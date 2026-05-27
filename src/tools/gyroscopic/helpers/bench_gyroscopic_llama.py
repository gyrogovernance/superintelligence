"""Gyroscopic LLM & QuBEC Climate Benchmark (Hardened Streaming Runner).

Native GyroLabe notes (for maintainers):
- gyrolabe_block_info_t.dq_lattice_empty: set in pack_DQ_lattice when defect D=B-P is all
  zeros; gyrolabe_qubec_matmul_q8_0 skips k4_gemv64_avx2 in that case (see gyrolabe_registry.h).
- GyroPulse footer lines are best-effort and may be missing if the process is killed on timeout (Windows).
- Matmul trace blocks use one stderr write so GyroRows and GyroPulse stay aligned under parallel ggml workers.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
import statistics

try:
    from src.tools.gyroscopic.config import (
        get_gyroscopic_llm_config,
        repo_root,
    )
    from src.tools.gyroscopic.constants import GYRO_ENV_VAR_NAMES
    from src.tools.gyroscopic.loader import build_llama_cli_command, run_llama_perplexity
    from src.tools.gyroscopic.ops_build import build_llama_cpp_if_needed
except ModuleNotFoundError:
    repo_root_path = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root_path))
    from src.tools.gyroscopic.config import (
        get_gyroscopic_llm_config,
        repo_root,
    )
    from src.tools.gyroscopic.constants import GYRO_ENV_VAR_NAMES
    from src.tools.gyroscopic.loader import build_llama_cli_command, run_llama_perplexity
    from src.tools.gyroscopic.ops_build import build_llama_cpp_if_needed

# === Robust Defaults (no CLI flags required) ===
# One llama prompt by default so a broken gyro path does not multiply wasted wall time.
PROMPT_SUITES: dict[str, list[str]] = {
    "smoke": [
        "Tell me about the Sun.",
    ],
    "diverse": [
        "Hello",
        "What is 17 multiplied by 19?",
        "Write a short Python function that reverses a linked list.",
        "Explain exact hybrid preservation in one sentence.",
    ],
    "extended": [
        "Hello",
        "What is 17 multiplied by 19?",
        "Write a short Python function that reverses a linked list.",
        "Explain exact hybrid preservation in one sentence.",
    ],
}
MULTI_CELL_MAX_CELLS = 8
# Distinct strings for native word4 batch only (not passed to llama-cli).
MULTI_CELL_NATIVE_LABELS = [f"cell-{i}" for i in range(MULTI_CELL_MAX_CELLS)]
# Perplexity and generation telemetry are noisy at short lengths; use a stable default.
DEFAULT_N_PREDICT = 32
# Cold GGUF load + CPU prompt eval often exceeds 120s on large models; override with --timeout.
TIMEOUT_STOCK = 600.0
TIMEOUT_GYRO = 1200.0
# Cold GGUF mmap + graph compile can go minutes with no pipe bytes; disable tracer
# collection on stock paths to reduce log noise and keep timing stable.
SILENT_KILL_SEC = 300.0  # Kill only if no stdout/stderr for this long (not wall-clock)
BENCH_SCOPE: list[str] = [
    "1. Theory and Architecture Compliance",
    "2. Valid High Quality Language Generation",
    "3. Better Performance than stock without bypassing Gyroscopic architecture because we can't figure out how to approach structure.",
]
ENABLE_HASH_MATCH = False

# Force CPU-safe flags. --no-context-shift prevents massive CPU reallocations.
# Bonsai-8B (Qwen3 base): suggested temp 0.5, top_k 20, top_p 0.85
EXTRA_ARGS = [
    "--seed", "42", "--temp", "0.5", "--top-p", "0.85", "--top-k", "20",
    "--reasoning", "off", "--single-turn",
    "--n-gpu-layers", "0", "--no-context-shift", "--flash-attn", "off",
    "--perf",
]
_INTEROP_SWEEP_DEFAULT_MODES = [
    "exact_substitution",
    "advisory",
    "approximate_derived",
]

_CLASS_NAMES = ("shell_radial", "shell_gauge", "chi_invariant", "chi_gauge", "generic")
_CLASS_DISPLAY_NAMES = ["shell-radial", "shell-x-gauge", "chi-invariant", "chi-x-gauge", "generic"]

_GYRO_ENV_PREFIXES = (
    "GGML_GYROSCOPIC",
    "GYROLABE_",
    "GYROGRAPH_",
    "GYRO_",
)


def _clean_llama_env(mode: str, *, trace: bool, witness_mode: str = "off") -> dict[str, str]:
    """Return a hermetic llama subprocess env for stock-vs-gyro comparisons.

    The parent shell often carries stale tuning variables from previous sweeps.
    Stock measurements must prove the compiled binary is quiescent, not merely
    that the active matmul hook declined at runtime.
    """
    env = os.environ.copy()
    for key in list(env):
        if any(key.startswith(prefix) for prefix in _GYRO_ENV_PREFIXES):
            env.pop(key, None)
    for key in GYRO_ENV_VAR_NAMES:
        env.pop(key, None)

    if mode == "gyroscopic":
        env["GGML_GYROSCOPIC"] = "1"
        env["GYROLABE_PARITY_WITNESS"] = witness_mode
        env["GGML_GYROSCOPIC_CLEAR_REGISTRY"] = "1"
        env["GGML_GYROSCOPIC_TRACE"] = "1" if trace else "0"
    else:
        env["GGML_GYROSCOPIC"] = "0"
        env["GGML_GYROSCOPIC_TRACE"] = "0"

    return env


def _zeros(arr_type: Any, n: int) -> Any:
    return (arr_type * n)(*(0 for _ in range(n)))

def _parse_interop_mode_list(raw: str) -> list[str]:
    modes: list[str] = []
    for item in (raw or "").split(","):
        s = item.strip().lower().replace("-", "_")
        if s:
            modes.append(s)
    return modes


def _build_interop_env(
    interop_mode: str | None,
    *,
    decode_max_batch_size: int | None,
    decode_max_chi_distance: int | None,
    decode_max_shell_delta: int | None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if interop_mode:
        env["GYROGRAPH_INTEROP_MODE"] = interop_mode
    if decode_max_batch_size is not None:
        env["GYROGRAPH_DECODE_MAX_BATCH_SIZE"] = str(int(decode_max_batch_size))
    if decode_max_chi_distance is not None:
        env["GYROGRAPH_DECODE_MAX_CHI_DISTANCE"] = str(int(decode_max_chi_distance))
    if decode_max_shell_delta is not None:
        env["GYROGRAPH_DECODE_MAX_SHELL_DELTA"] = str(int(decode_max_shell_delta))
    return env


DEFAULT_PERPLEXITY_REPEAT = 8
PERPLEXITY_CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In mathematics, a group is a set equipped with an operation that combines "
    "any two elements to form a third element while being associative as well as "
    "having an identity element and inverse elements. These three conditions, "

)

DATA_DIR = repo_root() / "data" / "benchmarks" / "gyroscopic_llama"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = DATA_DIR / "bench.json"

# === Trace Parsing ===
_TRACE_STATS_RE = re.compile(
    r"^(GyroMatMul stats|GyroRows|GyroRowsFidelity|GyroRowsComp|GyroRowsResidual|GyroDispatch|GyroGraph|GyroPlan|GyroExec|GyroCert|GyroKernel|GyroDirect|GyroPolicy|GyroDecode|GyroDecodeStats|GyroChiGaugeCache|GyroRegIndex|GyroSignatureQuery):\s*(.*)$"
)
# Per-call post-kernel summary emitted from ggml_gyroscopic backend.
_GYRO_POST_STATS_RE = re.compile(
    r"^GYRO_POST_STATS\s+(.+)$"
)
# Space-tolerant key=value pairs (split-on-whitespace loses "pq_rows = 1" style tokens).
_TRACE_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)")
_GYRO_REG_RE = re.compile(
    r"GYRO_REG:\s*tensors_scanned=(\d+)\s+q8_0_tensors=(\d+)\s+registry_entries=(\d+)"
    r"(?:\s+block_plan_hits=(\d+)\s+block_plan_misses=(\d+)\s+block_plan_predecoded_hits=(\d+))?"
)
_GYRO_PULSE_RE = re.compile(
    r"^GyroPulse:\s*attempts=(\d+)\s+qubec_calls=(\d+)\s+dense_calls=(\d+)\s+scanned_blocks=(\d+)\s+reg=(\d+)\s*$"
)
_REG_STATS_RE = re.compile(
    r"REG_STATS:.*?agg0=(\d+)\s+agg1=(\d+)\s+agg2=(\d+)\s+agg3=(\d+)\s+agg4=(\d+)"
)
_REG_STATS_SELECTED_RE = re.compile(
    r"REG_STATS:.*?sel0=(\d+)\s+sel1=(\d+)\s+sel2=(\d+)\s+sel3=(\d+)\s+sel4=(\d+)"
)
_REG_STATS_RESIDUAL_RE = re.compile(
    r"REG_STATS:.*?res0=(\d+)\s+res1=(\d+)\s+res2=(\d+)\s+res3=(\d+)"
)


_GYRO_PLAN_ALIASES: dict[str, list[str]] = {}
for _class_name in _CLASS_NAMES:
    _GYRO_PLAN_ALIASES[f"selected_{_class_name}_blocks"] = [
        f"selected_{_class_name}_blocks",
        f"projection_{_class_name}_blocks",
        f"projection_class_{_class_name}_blocks",
    ]
    _GYRO_PLAN_ALIASES[f"projection_{_class_name}_blocks"] = [
        f"projection_{_class_name}_blocks",
        f"projection_class_{_class_name}_blocks",
        f"selected_{_class_name}_blocks",
    ]
_GYRO_PLAN_ALIASES.update(
    {
        "residual_none_blocks": [
            "residual_none_blocks",
            "dq_none_blocks",
            "defect_none_blocks",
        ],
        "residual_k4_blocks": [
            "residual_k4_blocks",
            "dq_k4_blocks",
            "defect_k4_blocks",
        ],
        "residual_exact_q8_blocks": [
            "residual_exact_q8_blocks",
            "dq_q8_exact_blocks",
            "defect_exact_q8_blocks",
        ],
        "residual_backfill_debug_blocks": [
            "residual_backfill_debug_blocks",
            "dq_projected_backfill_debug_blocks",
            "defect_projectedbackfill_debug_blocks",
        ],
        "dq_none_blocks": ["dq_none_blocks", "defect_none_blocks", "residual_none_blocks"],
        "dq_k4_blocks": ["dq_k4_blocks", "defect_k4_blocks", "residual_k4_blocks"],
        "dq_q8_exact_blocks": [
            "dq_q8_exact_blocks",
            "defect_q8_exact_blocks",
            "residual_exact_q8_blocks",
        ],
        "dq_backfill_debug_blocks": [
            "dq_backfill_debug_blocks",
            "dq_projected_backfill_debug_blocks",
            "defect_projectedbackfill_debug_blocks",
            "residual_backfill_debug_blocks",
        ],
        "compiled_k4_residual_direct_decode_calls": [
            "compiled_k4_residual_direct_decode_calls",
        ],
        "compiled_k4_residual_blocks": ["compiled_k4_residual_blocks"],
        "kernel_generic_k4_blocks": ["kernel_generic_k4_blocks"],
        "exact_q8_generic_blocks": ["exact_q8_generic_blocks"],
        "direct_exact_generic_blocks": ["direct_exact_generic_blocks"],
        "chi_gauge_dq_kernel_calls": ["chi_gauge_dq_kernel_calls"],
        "chi_gauge_backfill_calls": ["chi_gauge_backfill_calls"],
    }
)

_GYRO_CERT_ALIASES: dict[str, list[str]] = {}
for _class_name in _CLASS_NAMES:
    _GYRO_CERT_ALIASES[f"certified_{_class_name}_blocks"] = [
        f"certified_{_class_name}_blocks",
        f"exact_containment_{_class_name}_blocks",
        f"containment_{_class_name}_blocks",
    ]
    _GYRO_CERT_ALIASES[f"exact_containment_{_class_name}_blocks"] = [
        f"exact_containment_{_class_name}_blocks",
        f"certified_{_class_name}_blocks",
        f"containment_{_class_name}_blocks",
    ]

_GYRO_EXEC_ALIASES: dict[str, list[str]] = {
    "residual_none_blocks": ["residual_none_blocks", "dq_none_blocks", "defect_none_blocks"],
    "residual_k4_blocks": ["residual_k4_blocks", "dq_k4_blocks", "defect_k4_blocks"],
    "residual_exact_q8_blocks": [
        "residual_exact_q8_blocks",
        "dq_q8_exact_blocks",
        "defect_exact_q8_blocks",
    ],
    "residual_backfill_debug_blocks": [
        "residual_backfill_debug_blocks",
        "dq_projected_backfill_debug_blocks",
        "dq_backfill_debug_blocks",
        "defect_projectedbackfill_debug_blocks",
    ],
    "compiled_k4_residual_direct_decode_calls": [
        "compiled_k4_residual_direct_decode_calls",
    ],
    "compiled_k4_residual_blocks": ["compiled_k4_residual_blocks"],
    "exact_q8_generic_blocks": ["exact_q8_generic_blocks"],
    "direct_exact_generic_blocks": ["direct_exact_generic_blocks"],
    "chi_gauge_dq_kernel_calls": ["chi_gauge_dq_kernel_calls"],
    "chi_gauge_backfill_calls": ["chi_gauge_backfill_calls"],
    "dq_none_blocks": ["dq_none_blocks", "residual_none_blocks", "defect_none_blocks"],
    "dq_k4_blocks": ["dq_k4_blocks", "residual_k4_blocks", "defect_k4_blocks"],
    "dq_q8_exact_blocks": [
        "dq_q8_exact_blocks",
        "residual_exact_q8_blocks",
        "defect_exact_q8_blocks",
    ],
    "dq_backfill_debug_blocks": [
        "dq_backfill_debug_blocks",
        "dq_projected_backfill_debug_blocks",
        "residual_backfill_debug_blocks",
        "defect_projectedbackfill_debug_blocks",
    ],
}

_GYRO_KERNEL_ALIASES = {
    **{
        f"exec_{_class_name}_blocks": [
            f"exec_{_class_name}_blocks",
            f"execution_{_class_name}_blocks",
        ]
        for _class_name in _CLASS_NAMES
    },
    **{
        "kernel_chi_gauge_k4_blocks": [
            "kernel_chi_gauge_k4_blocks",
            "kernel_pq_chi_gauge_dq_k4_blocks",
            "pq_chi_gauge__dq_k4",
        ],
        "kernel_chi_gauge_exact_q8_blocks": [
            "kernel_chi_gauge_exact_q8_blocks",
            "kernel_pq_chi_gauge_dq_q8_blocks",
            "pq_chi_gauge__dq_q8",
        ],
        "kernel_generic_exact_q8_blocks": [
            "kernel_generic_exact_q8_blocks",
            "kernel_pq_generic_dq_q8_blocks",
            "pq_generic__dq_q8",
        ],
        "kernel_generic_k4_blocks": [
            "kernel_generic_k4_blocks",
            "kernel_pq_generic_dq_k4_blocks",
            "pq_generic__dq_k4",
        ],
        "kernel_projected_backfill_blocks": [
            "kernel_projected_backfill_blocks",
            "kernel_projected_backfill",
        ],
        "exec_pq_chi_gauge_dq_k4_blocks": [
            "exec_pq_chi_gauge_dq_k4_blocks",
            "kernel_chi_gauge_k4_blocks",
            "kernel_pq_chi_gauge_dq_k4_blocks",
            "pq_chi_gauge__dq_k4",
        ],
        "exec_pq_chi_gauge_dq_q8_blocks": [
            "exec_pq_chi_gauge_dq_q8_blocks",
            "kernel_chi_gauge_exact_q8_blocks",
            "kernel_pq_chi_gauge_dq_q8_blocks",
            "pq_chi_gauge__dq_q8",
        ],
        "exec_pq_generic_dq_q8_blocks": [
            "exec_pq_generic_dq_q8_blocks",
            "kernel_generic_exact_q8_blocks",
            "kernel_pq_generic_dq_q8_blocks",
            "pq_generic__dq_q8",
        ],
        "exec_pq_generic_dq_k4_blocks": [
            "exec_pq_generic_dq_k4_blocks",
            "kernel_generic_k4_blocks",
            "kernel_pq_generic_dq_k4_blocks",
            "pq_generic__dq_k4",
        ],
    },
}


@dataclass
class _BlockClassCounts:
    shell_radial: int = 0
    shell_gauge: int = 0
    chi_invariant: int = 0
    chi_gauge: int = 0
    generic: int = 0


@dataclass
class TraceStats:
    qubec_calls: int = 0
    dense_calls: int = 0
    direct_exact_calls: int = 0
    structured_rows: int = 0
    dense_rows: int = 0
    direct_exact_rows: int = 0
    spectral_sparse_rows: int = 0
    pq_rows: int = 0
    dq_rows: int = 0
    pq_chi_rows: int = 0
    pq_shell_rows: int = 0
    dq_k4_rows: int = 0
    dq_q8_rows: int = 0
    residual_q8_rows: int = 0
    row_counts: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    dispatch_blocks: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    selected_blocks: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    projection_blocks: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    certified_blocks: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    exact_containment_blocks: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    exec_blocks: _BlockClassCounts = field(default_factory=_BlockClassCounts)
    attempt_rows: int = 0
    exact_witness: int = 0
    witness_sampled_rows: int = 0
    parity_mismatch: int = 0
    max_abs_row_error: float = 0.0
    dispatch_attempts: int = 0
    dispatch_scanned: int = 0
    no_structured_default_route: int = 0
    policy_residual_skipped_blocks: int = 0
    dispatch_k64_miss: int = 0
    registry_entries: int = 0
    exec_pq_chi_gauge_dq_k4_blocks: int = 0
    exec_pq_chi_gauge_dq_q8_blocks: int = 0
    exec_pq_generic_dq_q8_blocks: int = 0
    residual_none_blocks: int = 0
    residual_k4_blocks: int = 0
    residual_exact_q8_blocks: int = 0
    residual_backfill_debug_blocks: int = 0
    compiled_k4_residual_direct_decode_calls: int = 0
    dq_none_blocks: int = 0
    dq_k4_blocks: int = 0
    dq_q8_exact_blocks: int = 0
    dq_backfill_debug_blocks: int = 0
    compiled_k4_residual_blocks: int = 0
    exact_q8_generic_blocks: int = 0
    direct_exact_generic_blocks: int = 0
    chi_gauge_dq_kernel_calls: int = 0
    chi_gauge_backfill_calls: int = 0
    kernel_chi_gauge_k4_blocks: int = 0
    kernel_chi_gauge_exact_q8_blocks: int = 0
    kernel_generic_exact_q8_blocks: int = 0
    kernel_generic_k4_blocks: int = 0
    kernel_projected_backfill_blocks: int = 0
    pq_est_read_bytes: int = 0
    dq_est_read_bytes: int = 0
    dense_est_read_bytes: int = 0
    atlas_defect_error_sum: float = 0.0
    atlas_defect_error_max: float = 0.0
    atlas_defect_error_samples: int = 0
    witness_err_sum: float = 0.0
    witness_err_sq_sum: float = 0.0
    witness_err_samples: int = 0
    graph_m2_mean: float | None = None
    graph_cells: int = 0
    # Emitted at end of model load (stderr); present even if GyroDispatch footer never runs (timeout).
    reg_load_tensors_scanned: int = 0
    reg_load_q8_0_tensors: int = 0
    reg_load_registry_entries: int = 0
    reg_block_plan_hits: int = 0
    reg_block_plan_misses: int = 0
    reg_block_plan_predecoded_hits: int = 0
    # Last GyroPulse line (stderr) after each benchmark attempt.
    gyro_pulse_attempts: int = 0
    gyro_pulse_qubec_calls: int = 0
    gyro_pulse_dense_calls: int = 0
    gyro_pulse_scanned_blocks: int = 0
    gyro_pulse_reg: int = 0
    plan_load_path: str | None = None
    plan_load_rc: int | None = None
    plan_loaded: int = 0
    plan_entries: int = 0
    plan_bytes: int = 0
    interop_mode: str | None = None
    decode_max_batch_size: int = 0
    decode_max_chi_distance: int = 0
    decode_max_shell_delta: int = 0
    kv_eviction_threshold: float | None = None
    decode_cell: int = -1
    decode_chi6: int = -1
    decode_shell: int = -1
    decode_resonance_key: int = 0
    decode_chi_distance: int = 0
    decode_grouped_dispatch: int = 0
    decode_kv_priority: float | None = None
    decode_kv_score_norm: float | None = None
    decode_kv_evict: int = 0
    decode_m2_empirical: float | None = None
    decode_events: int = 0
    decode_grouped_dispatch_events: int = 0
    decode_ungrouped_dispatch_events: int = 0
    decode_kv_evict_events: int = 0
    decode_mean_chi_distance: float | None = None
    decode_max_chi_distance_seen: int = 0
    decode_mean_kv_priority: float | None = None
    chi_gauge_cache_hits: int = 0
    chi_gauge_cache_misses: int = 0
    chi_gauge_cache_stores: int = 0
    chi_gauge_cache_hit_rate: float | None = None
    reg_class_counts: list[int] | None = None
    reg_selected_projection_counts: list[int] | None = None
    reg_residual_format_counts: list[int] | None = None
    index_built_slots: int = 0
    index_total_entries: int = 0
    index_query_hits: int = 0
    index_query_misses: int = 0


def _block_alias_property(container: str, attr: str):
    def _get(self: "TraceStats") -> int:
        return int(getattr(getattr(self, container), attr))

    def _set(self: "TraceStats", value: int) -> None:
        setattr(getattr(self, container), attr, int(value))

    return property(_get, _set)


for _class_name in _CLASS_NAMES:
    setattr(TraceStats, f"{_class_name}_rows", _block_alias_property("row_counts", _class_name))
    setattr(TraceStats, f"selected_{_class_name}_blocks", _block_alias_property("selected_blocks", _class_name))
    setattr(TraceStats, f"projection_{_class_name}_blocks", _block_alias_property("projection_blocks", _class_name))
    setattr(TraceStats, f"certified_{_class_name}_blocks", _block_alias_property("certified_blocks", _class_name))
    setattr(TraceStats, f"exact_containment_{_class_name}_blocks", _block_alias_property("exact_containment_blocks", _class_name))
    setattr(TraceStats, f"exec_{_class_name}_blocks", _block_alias_property("exec_blocks", _class_name))

TraceStats.shell_radial_blocks = _block_alias_property("dispatch_blocks", "shell_radial")
TraceStats.shell_gauge_blocks = _block_alias_property("dispatch_blocks", "shell_gauge")
TraceStats.chi_invariant_blocks = _block_alias_property("dispatch_blocks", "chi_invariant")
TraceStats.chi_gauge_blocks = _block_alias_property("dispatch_blocks", "chi_gauge")
TraceStats.generic_blocks = _block_alias_property("dispatch_blocks", "generic")
TraceStats.residual_q8_blocks = property(
    fget=lambda self: getattr(self, "residual_q8_rows", 0),
    fset=lambda self, value: setattr(self, "residual_q8_rows", int(value)),
)

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
    llama_prompt_eval_ms: float | None = None
    llama_prompt_eval_tokens: int | None = None
    llama_gen_eval_ms: float | None = None
    llama_gen_eval_tokens: int | None = None
    wall_effective_gen_tps: float | None = None
    interop_mode: str | None = None
    silent_kill: bool = False
    llama_cli_fingerprint: str | None = None

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.silent_kill and self.rc == 0

    @property
    def status_label(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.silent_kill:
            return "silent_kill"
        if self.rc is None:
            return "not_run"
        if self.rc != 0:
            return f"fail_rc_{self.rc}"
        return "ok"


def _format_run_status(result: RunResult) -> str:
    if result.timed_out:
        return "TIMEOUT"
    if result.silent_kill:
        return "KILLED"
    if result.rc != 0:
        return f"FAIL({result.rc})"
    return "OK"


def run_measurement_exposure(r: RunResult) -> dict[str, Any]:
    """Expose wall vs llama-reported perf vs Gyro trace telemetry in one JSON-friendly dict."""
    tr = r.trace
    row_sum = int(tr.structured_rows) + int(tr.dense_rows) + int(tr.direct_exact_rows)
    struct_frac = (float(tr.structured_rows) / float(row_sum)) if row_sum > 0 else None
    direct_frac = (float(tr.direct_exact_rows) / float(row_sum)) if row_sum > 0 else None
    est_total = int(tr.pq_est_read_bytes) + int(tr.dq_est_read_bytes) + int(tr.dense_est_read_bytes)
    pq_share = (float(tr.pq_est_read_bytes) / float(est_total)) if est_total > 0 else None
    return {
        "wall_seconds": r.elapsed,
        "llama_reported_prompt_tps": r.prompt_tps,
        "llama_reported_gen_tps": r.gen_tps,
        "llama_prompt_eval_ms": r.llama_prompt_eval_ms,
        "llama_prompt_eval_tokens": r.llama_prompt_eval_tokens,
        "llama_gen_eval_ms": r.llama_gen_eval_ms,
        "llama_gen_eval_tokens": r.llama_gen_eval_tokens,
        "wall_effective_gen_tps": r.wall_effective_gen_tps,
        "trace_structured_rows": tr.structured_rows,
        "trace_dense_rows": tr.dense_rows,
        "trace_direct_exact_rows": tr.direct_exact_rows,
        "trace_structured_row_fraction": struct_frac,
        "trace_direct_exact_row_fraction": direct_frac,
        "trace_spectral_sparse_rows": tr.spectral_sparse_rows,
        "trace_execution_route_stock_rows": max(0, int(tr.dense_rows) - int(tr.spectral_sparse_rows)),
        "trace_execution_route_spectral_sparse_rows": tr.spectral_sparse_rows,
        "trace_execution_route_native_rows": tr.structured_rows,
        "trace_pq_chi_rows": tr.pq_chi_rows,
        "trace_pq_shell_rows": tr.pq_shell_rows,
        "trace_dq_k4_rows": tr.dq_k4_rows,
        "trace_dq_q8_rows": tr.dq_q8_rows,
        "trace_residual_q8_blocks": tr.residual_q8_blocks,
        "trace_generic_blocks": tr.generic_blocks,
        "trace_pq_est_read_bytes": tr.pq_est_read_bytes,
        "trace_dq_est_read_bytes": tr.dq_est_read_bytes,
        "trace_dense_est_read_bytes": tr.dense_est_read_bytes,
        "trace_est_read_bytes_total": est_total,
        "trace_pq_share_of_est_reads": pq_share,
    }


def _is_zeroish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return False
    if isinstance(value, float):
        return False
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple)):
        return all(_is_zeroish(v) for v in value)
    if isinstance(value, dict):
        return all(_is_zeroish(v) for v in value.values())
    return False


def _compact_metrics(payload: Any) -> Any:
    if isinstance(payload, dict):
        compact: dict[str, Any] = {}
        for key, value in payload.items():
            compacted_value = _compact_metrics(value)
            if compacted_value is None:
                continue
            compact[key] = compacted_value
        return compact if compact else None
    if isinstance(payload, list):
        if not payload:
            return None
        if all(_is_zeroish(v) for v in payload):
            return None
        if all(isinstance(v, (bool, int, float, str, type(None))) for v in payload):
            return list(payload)
        compacted_items: list[Any] = []
        for item in payload:
            compacted_item = _compact_metrics(item)
            if compacted_item is None:
                continue
            compacted_items.append(compacted_item)
        return compacted_items if compacted_items else None
    if _is_zeroish(payload):
        return None
    return payload


def _compact_trace(r: RunResult) -> dict[str, Any]:
    return _compact_metrics(asdict(r.trace)) or {}


def _compact_measurement_exposure(r: RunResult) -> dict[str, Any]:
    return _compact_metrics(run_measurement_exposure(r)) or {}


MEASUREMENT_LEGEND: dict[str, str] = {
    "wall_seconds": (
        "Host wall time for the entire llama-cli subprocess (model load, prompt, decode, trace I/O, exit)."
    ),
    "llama_prompt_eval_ms": (
        "Prompt eval wall-clock milliseconds extracted from llama-cli perf output (ms / token_count)."
    ),
    "llama_prompt_eval_tokens": (
        "Prompt eval token count used with prompt_eval_ms to compute prompt_eval-based TPS when needed."
    ),
    "llama_gen_eval_ms": (
        "Generation eval wall-clock milliseconds extracted from llama-cli perf output (ms / token_count)."
    ),
    "llama_gen_eval_tokens": (
        "Generation eval token count used for wall_effective_gen_tps and sanity checks."
    ),
    "llama_reported_gen_tps": (
        "Tokens per second implied by llama-cli perf text (bracket line or eval time ms / tokens). "
        "This is llama's generation-eval chart, not wall_seconds^-1 * tokens."
    ),
    "wall_effective_gen_tps": (
        "llama_gen_eval_tokens / wall_seconds when both are known: same token count as llama's gen eval, "
        "divided by full subprocess wall. Compare to llama_reported_gen_tps to see load-vs-decode split."
    ),
    "trace_est_read_bytes_model": (
        "Telemetry uses GyroLabe design constants 512 vs 16384 bytes per row-classified row "
        "(gyrolabe_kernel_qubec_matmul.c GYROLABE_EST_PQ_READ_BYTES_PER_ROW and GYROLABE_EST_DENSE_READ_BYTES_PER_ROW); "
        "not hardware-counter measured DRAM."
    ),
    "trace_spectral_sparse_rows": (
        "Rows that matched the spectral-sparse execution route during trace (GyroRows section)."
    ),
    "trace_execution_route_stock_rows": (
        "Rows executed via STOCK path. Derived as dense_rows - spectral_sparse_rows."
    ),
    "trace_execution_route_spectral_sparse_rows": (
        "Rows executed via SPECTRAL_SPARSE route."
    ),
    "trace_execution_route_native_rows": (
        "Rows executed via NATIVE route."
    ),
    "trace_pq_chi_rows": (
        "P_Q rows attributed to chi-structured decomposition."
    ),
    "trace_pq_shell_rows": (
        "P_Q rows attributed to shell-structured decomposition."
    ),
    "trace_dq_k4_rows": (
        "D_Q rows using K4-packed residual decomposition."
    ),
    "trace_dq_q8_rows": (
        "D_Q rows using exact Q8 residual blocks."
    ),
    "trace_generic_blocks": (
        "Registry-generic blocks discovered by dispatch, used for baseline pressure/coverage reporting."
    ),
    "theory_note": (
        "aQPU two-step uniformization and future-cone entropy on Omega are kernel-byte phenomena; "
        "they justify native spectral composition in QuBEC climate math, not a literal claim that "
        "each transformer token advances Omega by two bytes."
    ),
}

def _parse_gyro_trace_kv(rest: str) -> dict[str, str]:
    return {m.group(1): m.group(2).rstrip(",") for m in _TRACE_KV_RE.finditer(rest)}


def _kv_get_int(kv: dict[str, str], *keys: str, default: int = 0) -> int:
    for k in keys:
        v = kv.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return default


def _kv_get_float(kv: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = kv.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return default


def _set_from_aliases(tr: TraceStats, kv: dict[str, str], aliases: dict[str, list[str]]) -> None:
    for field, keys in aliases.items():
        setattr(tr, field, _kv_get_int(kv, *keys))


def _parse_trace(stderr_lines: list[str]) -> TraceStats:
    tr = TraceStats()
    for line in stderr_lines:
        s = line.strip()
        mreg = _GYRO_REG_RE.search(s)
        if mreg:
            tr.reg_load_tensors_scanned = int(mreg.group(1))
            tr.reg_load_q8_0_tensors = int(mreg.group(2))
            tr.reg_load_registry_entries = int(mreg.group(3))
            if len(mreg.groups()) >= 6:
                tr.reg_block_plan_hits = int(mreg.group(4) or 0)
                tr.reg_block_plan_misses = int(mreg.group(5) or 0)
                tr.reg_block_plan_predecoded_hits = int(mreg.group(6) or 0)
            continue
        mp = _GYRO_PULSE_RE.match(s)
        if mp:
            tr.gyro_pulse_attempts = int(mp.group(1))
            tr.gyro_pulse_qubec_calls = int(mp.group(2))
            tr.gyro_pulse_dense_calls = int(mp.group(3))
            tr.gyro_pulse_scanned_blocks = int(mp.group(4))
            tr.gyro_pulse_reg = int(mp.group(5))
            continue
        if s.startswith("GYRO_PLAN_LOAD:"):
            kv = _parse_gyro_trace_kv(s)
            tr.plan_load_path = kv.get("path")
            tr.plan_load_rc = int(kv.get("rc", -1))
            tr.plan_loaded = int(kv.get("loaded", 0))
            tr.plan_entries = int(kv.get("entries", 0))
            tr.plan_bytes = int(kv.get("bytes", 0))
            continue
        mreg_stats = _REG_STATS_RE.search(s)
        if mreg_stats:
            tr.reg_class_counts = [int(mreg_stats.group(i)) for i in range(1, 6)]
        mreg_sel = _REG_STATS_SELECTED_RE.search(s)
        if mreg_sel:
            tr.reg_selected_projection_counts = [int(mreg_sel.group(i)) for i in range(1, 6)]
        mreg_res = _REG_STATS_RESIDUAL_RE.search(s)
        if mreg_res:
            tr.reg_residual_format_counts = [int(mreg_res.group(i)) for i in range(1, 5)]
        if mreg_stats or mreg_sel or mreg_res:
            continue
        mpost = _GYRO_POST_STATS_RE.match(s)
        if mpost:
            post_kv = _parse_gyro_trace_kv(mpost.group(1))
            tr.structured_rows = int(post_kv.get("structured", tr.structured_rows))
            tr.dense_rows = int(post_kv.get("dense", tr.dense_rows))
            tr.direct_exact_rows = int(post_kv.get("direct", tr.direct_exact_rows))
            if "generic" in post_kv:
                tr.row_counts.generic = int(post_kv.get("generic", 0))
            tr.pq_rows = int(post_kv.get("pq", tr.pq_rows))
            tr.dq_rows = int(post_kv.get("dq", tr.dq_rows))
            tr.no_structured_default_route = int(
                post_kv.get("no_structured_default_route", tr.no_structured_default_route)
            )
            tr.direct_exact_generic_blocks = int(post_kv.get("generic_exact", tr.direct_exact_generic_blocks))
            tr.kernel_generic_exact_q8_blocks = int(
                post_kv.get("kernel_generic_q8", tr.kernel_generic_exact_q8_blocks)
            )
            tr.qubec_calls = tr.pq_rows
            tr.dense_calls = tr.dense_rows
            continue
        m = _TRACE_STATS_RE.match(s)
        if not m: continue
        section, rest = m.group(1), m.group(2)
        kv = _parse_gyro_trace_kv(rest)
        if section == "GyroMatMul stats":
            pq = int(kv.get("pq_rows", kv.get("qubec_calls", 0)))
            dr = int(kv.get("dense_rows", kv.get("dense_calls", 0)))
            tr.qubec_calls = pq
            tr.dense_calls = dr
            tr.pq_rows = pq
        elif section == "GyroRows":
            tr.structured_rows = _kv_get_int(kv, "structured_rows", default=0)
            tr.dense_rows = _kv_get_int(kv, "dense_rows", default=0)
            tr.direct_exact_rows = _kv_get_int(kv, "direct_exact_rows", default=0)
            tr.spectral_sparse_rows = _kv_get_int(kv, "spectral_sparse_rows", default=0)
            tr.pq_rows = _kv_get_int(kv, "pq_rows", default=0)
            tr.dq_rows = _kv_get_int(kv, "dq_rows", default=0)
            tr.shell_radial_rows = _kv_get_int(kv, "shell_radial_rows", default=0)
            tr.shell_gauge_rows = _kv_get_int(kv, "shell_gauge_rows", default=0)
            tr.chi_invariant_rows = _kv_get_int(kv, "chi_invariant_rows", default=0)
            tr.chi_gauge_rows = _kv_get_int(kv, "chi_gauge_rows", default=0)
            tr.generic_rows = _kv_get_int(kv, "generic_rows", default=0)
            tr.attempt_rows = _kv_get_int(kv, "structured_attempt_rows", default=0)
            tr.exact_witness = _kv_get_int(kv, "exact_witness_rows", default=0)
            tr.witness_sampled_rows = _kv_get_int(kv, "witness_sampled_rows", default=0)
            tr.parity_mismatch = _kv_get_int(kv, "parity_mismatch_rows", default=0)
            tr.max_abs_row_error = _kv_get_float(kv, "max_abs_row_error", default=0.0)
            tr.pq_est_read_bytes = _kv_get_int(kv, "pq_est_read_bytes", default=0)
            tr.dq_est_read_bytes = _kv_get_int(kv, "dq_est_read_bytes", default=0)
            tr.dense_est_read_bytes = _kv_get_int(kv, "dense_est_read_bytes", default=0)
            tr.qubec_calls = tr.pq_rows
            tr.dense_calls = tr.dense_rows
        elif section == "GyroRowsComp":
            tr.pq_chi_rows = _kv_get_int(
                kv,
                "pq_chi_rows",
                default=_kv_get_int(kv, "chi_rows", default=0),
            )
            tr.pq_shell_rows = _kv_get_int(
                kv,
                "pq_shell_rows",
                default=_kv_get_int(kv, "shell_rows", default=0),
            )
            tr.dq_k4_rows = _kv_get_int(kv, "dq_k4_rows", default=0)
            tr.dq_q8_rows = _kv_get_int(
                kv,
                "dq_q8_rows",
                default=_kv_get_int(kv, "defect_q8_rows", default=0),
            )
        elif section == "GyroRowsFidelity":
            tr.atlas_defect_error_sum = _kv_get_float(kv, "atlas_defect_error_sum", default=0.0)
            tr.atlas_defect_error_max = _kv_get_float(kv, "atlas_defect_error_max", default=0.0)
            tr.atlas_defect_error_samples = _kv_get_int(kv, "atlas_defect_error_samples", default=0)
            tr.witness_err_sum = _kv_get_float(kv, "witness_err_sum", default=0.0)
            tr.witness_err_sq_sum = _kv_get_float(kv, "witness_err_sq_sum", default=0.0)
            tr.witness_err_samples = _kv_get_int(kv, "witness_err_samples", default=0)
        elif section == "GyroRowsResidual":
            tr.residual_q8_rows = _kv_get_int(kv, "residual_q8_rows", default=tr.residual_q8_rows)
        elif section == "GyroDispatch":
            tr.dispatch_attempts = _kv_get_int(kv, "attempts", default=0)
            tr.no_structured_default_route = _kv_get_int(
                kv,
                "no_structured_default_route",
                default=0,
            )
            tr.policy_residual_skipped_blocks = _kv_get_int(
                kv,
                "policy_residual_skipped",
                default=0,
            )
            tr.dispatch_scanned = _kv_get_int(kv, "scanned_blocks", default=0)
            tr.dispatch_k64_miss = _kv_get_int(kv, "no_k64_blocks", default=0)
            tr.registry_entries = _kv_get_int(kv, "dispatch_entries", default=0)
            tr.shell_radial_blocks = _kv_get_int(kv, "shell_radial_blocks", default=0)
            tr.shell_gauge_blocks = _kv_get_int(kv, "shell_gauge_blocks", default=0)
            tr.chi_invariant_blocks = _kv_get_int(kv, "chi_invariant_blocks", default=0)
            tr.chi_gauge_blocks = _kv_get_int(kv, "chi_gauge_blocks", default=0)
            tr.generic_blocks = _kv_get_int(kv, "generic_blocks", default=0)
            tr.index_query_hits = _kv_get_int(kv, "index_query_hits", default=0)
            tr.index_query_misses = _kv_get_int(kv, "index_query_misses", default=0)
        elif section == "GyroRegIndex":
            tr.index_built_slots = int(kv.get("built_slots", 0))
            tr.index_total_entries = int(kv.get("total_entries", 0))
        elif section == "GyroPlan":
            _set_from_aliases(tr, kv, _GYRO_PLAN_ALIASES)
        elif section == "GyroCert":
            _set_from_aliases(tr, kv, _GYRO_CERT_ALIASES)
        elif section == "GyroExec":
            for field, keys in _GYRO_EXEC_ALIASES.items():
                if getattr(tr, field, 0) == 0:
                    setattr(tr, field, _kv_get_int(kv, *keys, default=0))
        elif section == "GyroKernel":
            _set_from_aliases(tr, kv, _GYRO_KERNEL_ALIASES)
        elif section == "GyroDirect":
            tr.direct_exact_calls = _kv_get_int(kv, "calls", default=0)
            tr.direct_exact_rows = _kv_get_int(kv, "rows", default=tr.direct_exact_rows)
            tr.direct_exact_generic_blocks = _kv_get_int(
                kv,
                "generic_exact_blocks",
                "direct_exact_generic_blocks",
                default=tr.direct_exact_generic_blocks,
            )
        elif section == "GyroGraph":
            tr.graph_m2_mean = _kv_get_float(kv, "m2_mean", default=0.0)
            tr.graph_cells = _kv_get_int(kv, "cells", default=0)
        elif section == "GyroPolicy":
            tr.interop_mode = kv.get("interop_mode")
            tr.decode_max_batch_size = _kv_get_int(kv, "decode_max_batch_size", default=0)
            tr.decode_max_chi_distance = _kv_get_int(kv, "decode_max_chi_distance", default=0)
            tr.decode_max_shell_delta = _kv_get_int(kv, "decode_max_shell_delta", default=0)
            tr.kv_eviction_threshold = _kv_get_float(kv, "kv_eviction_threshold", default=0.0)
        elif section == "GyroDecode":
            tr.decode_cell = _kv_get_int(kv, "cell", default=-1)
            tr.decode_chi6 = _kv_get_int(kv, "chi6", default=-1)
            tr.decode_shell = _kv_get_int(kv, "shell", default=-1)
            tr.decode_resonance_key = _kv_get_int(kv, "resonance_key", default=0)
            tr.decode_chi_distance = _kv_get_int(kv, "chi_distance", default=0)
            tr.decode_grouped_dispatch = _kv_get_int(kv, "grouped_dispatch", default=0)
            tr.decode_kv_priority = _kv_get_float(kv, "kv_priority", default=0.0)
            tr.decode_kv_score_norm = _kv_get_float(kv, "kv_score_norm", default=0.0)
            tr.decode_kv_evict = _kv_get_int(kv, "kv_evict", default=0)
            tr.decode_m2_empirical = _kv_get_float(kv, "m2_empirical", default=0.0)
        elif section == "GyroDecodeStats":
            tr.decode_events = _kv_get_int(kv, "events", default=0)
            tr.decode_grouped_dispatch_events = _kv_get_int(kv, "grouped_dispatch", default=0)
            tr.decode_ungrouped_dispatch_events = _kv_get_int(kv, "ungrouped_dispatch", default=0)
            tr.decode_kv_evict_events = _kv_get_int(kv, "kv_evict", default=0)
            tr.decode_mean_chi_distance = _kv_get_float(kv, "mean_chi_distance", default=0.0)
            tr.decode_max_chi_distance_seen = _kv_get_int(kv, "max_chi_distance", default=0)
            tr.decode_mean_kv_priority = _kv_get_float(kv, "mean_kv_priority", default=0.0)
        elif section == "GyroChiGaugeCache":
            tr.chi_gauge_cache_hits = _kv_get_int(kv, "hits", default=0)
            tr.chi_gauge_cache_misses = _kv_get_int(kv, "misses", default=0)
            tr.chi_gauge_cache_stores = _kv_get_int(kv, "stores", default=0)
            tr.chi_gauge_cache_hit_rate = _kv_get_float(kv, "hit_rate", default=0.0)
    if tr.registry_entries == 0:
        tr.registry_entries = tr.reg_load_q8_0_tensors
    if tr.dispatch_attempts and tr.decode_events == 0:
        tr.decode_events = tr.dispatch_attempts
        tr.decode_grouped_dispatch_events = tr.decode_grouped_dispatch_events or tr.dispatch_attempts
    return tr


_THROUGHPUT_BRACKET_RE = re.compile(
    r"\[\s*Prompt:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*t/s\s*\|\s*"
    r"Generation:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*t/s\s*\]"
)
_TPS_TAIL_RE = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+tokens per second"
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
class LlamaPerfExtract:
    """Parsed llama-cli perf: throughput figures and raw eval windows when present."""

    prompt_tps: float | None = None
    gen_tps: float | None = None
    prompt_eval_ms: float | None = None
    prompt_eval_tokens: int | None = None
    gen_eval_ms: float | None = None
    gen_eval_tokens: int | None = None


def _normalize_cli_log_text(text: str) -> str:
    if not text:
        return ""
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _fill_llama_perf_from_text(ex: LlamaPerfExtract, combined: str) -> None:
    m = _THROUGHPUT_BRACKET_RE.search(combined)
    if m:
        try:
            ex.prompt_tps = float(m.group(1))
            ex.gen_tps = float(m.group(2))
        except ValueError:
            pass

    for line in combined.splitlines():
        mp = _PERF_PROMPT_MS_RE.search(line)
        if mp:
            ms, n = float(mp.group(1)), max(int(mp.group(2)), 1)
            if ms > 0:
                ex.prompt_tps = 1000.0 * n / ms
                ex.prompt_eval_ms = ms
                ex.prompt_eval_tokens = n
        mg = _PERF_GEN_MS_RE.search(line)
        if mg:
            ms, n = float(mg.group(1)), max(int(mg.group(2)), 1)
            if ms > 0:
                ex.gen_tps = 1000.0 * n / ms
                ex.gen_eval_ms = ms
                ex.gen_eval_tokens = n

    if ex.prompt_tps is None or ex.gen_tps is None:
        for line in combined.splitlines():
            low = line.lower()
            if "prompt eval time" in low and "tokens per second" in low:
                m2 = _TPS_TAIL_RE.search(line)
                if m2:
                    try:
                        ex.prompt_tps = float(m2.group(1))
                    except ValueError:
                        pass
            elif "prompt eval time" not in low and "tokens per second" in low:
                m2 = _TPS_TAIL_RE.search(line)
                if m2:
                    try:
                        ex.gen_tps = float(m2.group(1))
                    except ValueError:
                        pass


def parse_llama_perf_extract(stdout: str, stderr: str) -> LlamaPerfExtract:
    """
    Parse llama-cli perf text. Prefer stdout-only when both prompt and gen t/s are
    present there (same rule as parse_llama_throughput).
    """
    out = _normalize_cli_log_text(stdout)
    err = _normalize_cli_log_text(stderr)
    full = out + "\n" + err
    ex_out = LlamaPerfExtract()
    _fill_llama_perf_from_text(ex_out, out)
    ex_full = LlamaPerfExtract()
    _fill_llama_perf_from_text(ex_full, full)
    if ex_out.prompt_tps is not None and ex_out.gen_tps is not None:
        if ex_out.prompt_eval_ms is None:
            ex_out.prompt_eval_ms = ex_full.prompt_eval_ms
        if ex_out.prompt_eval_tokens is None:
            ex_out.prompt_eval_tokens = ex_full.prompt_eval_tokens
        if ex_out.gen_eval_ms is None:
            ex_out.gen_eval_ms = ex_full.gen_eval_ms
        if ex_out.gen_eval_tokens is None:
            ex_out.gen_eval_tokens = ex_full.gen_eval_tokens
        return ex_out
    if ex_out.prompt_tps is None:
        ex_out.prompt_tps = ex_full.prompt_tps
    if ex_out.gen_tps is None:
        ex_out.gen_tps = ex_full.gen_tps
    if ex_out.prompt_eval_ms is None:
        ex_out.prompt_eval_ms = ex_full.prompt_eval_ms
    if ex_out.prompt_eval_tokens is None:
        ex_out.prompt_eval_tokens = ex_full.prompt_eval_tokens
    if ex_out.gen_eval_ms is None:
        ex_out.gen_eval_ms = ex_full.gen_eval_ms
    if ex_out.gen_eval_tokens is None:
        ex_out.gen_eval_tokens = ex_full.gen_eval_tokens
    return ex_out


def parse_llama_throughput(stdout: str, stderr: str) -> tuple[float | None, float | None]:
    """
    llama_perf_context_print uses stderr (tokens per second). Some builds print
    a one-line [ Prompt: x t/s | Generation: y t/s ] summary on stdout.
    Prefer parsing stdout alone when it is complete so huge Gyro trace lines on
    stderr cannot confuse later regex passes.
    """
    ex = parse_llama_perf_extract(stdout, stderr)
    return ex.prompt_tps, ex.gen_tps


def _print_section(title: str, items: list[tuple[str, Any]], *, width: int = 16) -> None:
    if not any(v for _, v in items):
        return
    print(f"\n--- {title} ---\n")
    for name, value in items:
        print(f"  {name:<{width}} {value:>8}")


def _print_distribution(
    title: str,
    names: list[str],
    counts: list[int] | None,
    *,
    extra_total: int | None = None,
) -> None:
    if counts is None:
        return
    print(f"\n--- {title} ---\n")
    total = sum(counts) if counts is not None else 0
    if extra_total is not None:
        total = extra_total
    for i, name in enumerate(names):
        value = counts[i] if i < len(counts) else 0
        pct = 100.0 * value / total if total > 0 else 0.0
        print(f"  {name:<16} {value:>8}  ({pct:.1f}%)")

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
    witness_mode: str = "off",
    interop_mode: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> RunResult:
    cfg = get_gyroscopic_llm_config()
    env = _clean_llama_env(mode, trace=trace, witness_mode=witness_mode)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    
    args = build_llama_cli_command(cfg, prompt=prompt, n_predict=n_predict, extra_args=EXTRA_ARGS)

    llama_cli_fingerprint: str | None = None
    try:
        exe = Path(args[0]).resolve()
        if exe.is_file():
            st = exe.stat()
            llama_cli_fingerprint = f"{int(st.st_mtime_ns)}_{int(st.st_size)}"
    except OSError:
        llama_cli_fingerprint = None

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
        if time.perf_counter() - last_read[0] > SILENT_KILL_SEC:
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
    perf = parse_llama_perf_extract(out_text, err_text)
    gen_text = _extract_generation(stdout_lines)
    elapsed_final = time.perf_counter() - start
    wall_eff: float | None = None
    if perf.gen_eval_tokens is not None and perf.gen_eval_tokens > 0 and elapsed_final > 0.0:
        wall_eff = float(perf.gen_eval_tokens) / float(elapsed_final)

    return RunResult(
        mode=mode,
        prompt_idx=idx,
        prompt=prompt,
        elapsed=elapsed_final,
        timed_out=timed_out,
        rc=proc.returncode,
        prompt_tps=perf.prompt_tps,
        gen_tps=perf.gen_tps,
        gen_text=gen_text,
        trace=_parse_trace(stderr_lines),
        stdout_hash=hashlib.sha256(gen_text.encode()).hexdigest()[:12],
        llama_prompt_eval_ms=perf.prompt_eval_ms,
        llama_prompt_eval_tokens=perf.prompt_eval_tokens,
        llama_gen_eval_ms=perf.gen_eval_ms,
        llama_gen_eval_tokens=perf.gen_eval_tokens,
        wall_effective_gen_tps=wall_eff,
        interop_mode=interop_mode,
        silent_kill=silent_kill,
        llama_cli_fingerprint=llama_cli_fingerprint,
    )

# === Native Climate Probe ===
def run_kv_priority_illustrative() -> dict[str, Any]:
    """
    Illustrative climate ranking only (not wired to llama KV yet).
    Lower score => more eviction pressure in this toy ordering.
    """
    from src.tools.gyroscopic.gyrograph_runtime import cell_climate_from_histograms

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
            CellBuffers,
            gyrograph_emit_slcp_batch,
            gyrograph_ingest_word4_batch_indexed,
            gyromatmul_runtime_caps,
        )
        from src.tools.gyroscopic.gyrograph_runtime import (
            kv_eviction_priority_from_slcp,
            plan_decode_groups,
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
    omega12 = _zeros(ct.c_int32, n)
    step = _zeros(ct.c_uint64, n)
    last_byte = _zeros(ct.c_uint8, n)
    has_closed = _zeros(ct.c_uint8, n)
    word4 = (ct.c_uint8 * (4 * n))(*(0 for _ in range(4 * n)))
    chi_ring = (ct.c_uint8 * (64 * n))(*(0 for _ in range(64 * n)))
    chi_pos = _zeros(ct.c_uint8, n)
    chi_valid = _zeros(ct.c_uint8, n)
    chi_hist = (ct.c_uint16 * (64 * n))(*(0 for _ in range(64 * n)))
    shell_hist = (ct.c_uint16 * (7 * n))(*(0 for _ in range(7 * n)))
    fam_ring = (ct.c_uint8 * (64 * n))(*(0 for _ in range(64 * n)))
    fam_hist = (ct.c_uint16 * (4 * n))(*(0 for _ in range(4 * n)))
    omega_sig = _zeros(ct.c_int32, n)
    p_o = _zeros(ct.c_uint16, n)
    p_e = _zeros(ct.c_uint16, n)
    pbit = _zeros(ct.c_uint8, n)
    words_in = (ct.c_uint8 * (4 * n))()
    for i, text in enumerate(prompts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for b in range(4):
            words_in[4 * i + b] = digest[b]
    rkey = _zeros(ct.c_uint32, n)

    t0 = time.perf_counter()
    ing_bufs = CellBuffers(
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
        rkey,
    )
    gyrograph_ingest_word4_batch_indexed(
        ing_bufs,
        cell_ids,
        words_in,
        profile,
        n,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    slcp_bufs = CellBuffers(
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
        rkey,
    )
    batch = gyrograph_emit_slcp_batch(
        n,
        cell_ids,
        slcp_bufs,
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
    decode_groups = plan_decode_groups(batch, max_batch_size=max(1, min(8, n)))
    kv_scores = [kv_eviction_priority_from_slcp(s) for s in batch]

    return {
        "cells": n,
        "ingest_elapsed_ms": elapsed_ms,
        "unique_chi6": uniq_chi6,
        "grouping_metric": grouping,
        "m2_empirical_mean": statistics.mean(m2_vals) if m2_vals else None,
        "m2_empirical_stdev": statistics.stdev(m2_vals) if len(m2_vals) > 1 else None,
        "steps_ok": all(int(batch[i].step) == 4 for i in range(n)),
        "decode_group_count": len(decode_groups),
        "decode_group_sizes": [len(g) for g in decode_groups],
        "kv_priority_min": min(kv_scores) if kv_scores else None,
        "kv_priority_mean": statistics.mean(kv_scores) if kv_scores else None,
        "kv_priority_max": max(kv_scores) if kv_scores else None,
    }


def run_qubec_climate_probe() -> dict[str, Any]:
    out: dict[str, Any] = {"status": "pass", "details": {}}
    try:
        from src.tools.gyroscopic.ops import (
            gyrolabe_analyze_operator_64, gyrolabe_chirality_evolve_n,
            CellBuffers,
            gyrograph_emit_slcp_batch,
            gyromatmul_runtime_caps,
        )
        gyromatmul_runtime_caps()
        hist, ens = [0]*64, [1]*64; hist[0] = 64
        chi = gyrolabe_chirality_evolve_n(hist, ens, 2)
        out["details"]["chirality_2step_uniform"] = sum(chi)==64 and max(chi)-min(chi)<=1
        
        import numpy as np

        def _chi_gauge_dense_from_spatial_kernels(subkernels: np.ndarray) -> np.ndarray:
            W = np.zeros((64, 64), dtype=np.float32)
            for go in range(4):
                for gi in range(4):
                    kernel = subkernels[go, gi]
                    for r in range(16):
                        for c in range(16):
                            W[go * 16 + r, gi * 16 + c] = kernel[c ^ r]
            return W

        subkernels = np.zeros((4, 4, 16), dtype=np.float32)
        for go in range(4):
            for gi in range(4):
                base = (go + 1) * 7 + (gi + 1) * 3
                subkernels[go, gi] = np.array(
                    [base + ((h * go + 2 * gi + h) % 5) - 2 for h in range(16)],
                    dtype=np.float32,
                )
        wr = _chi_gauge_dense_from_spatial_kernels(subkernels)
        rep = gyrolabe_analyze_operator_64(wr, threshold=0.01)
        op_cls = int(rep.op_class)
        out["details"]["probe_operator_class"] = op_cls
        out["details"]["operator_report_fields_ok"] = bool(
            hasattr(rep, "op_class")
            and hasattr(rep, "proj_energy_ratio")
            and hasattr(rep, "defect_norm")
            and hasattr(rep, "eigenvalues_valid")
        )
        out["details"]["hybrid_routing_ready"] = (
            op_cls != 0 and int(rep.eigenvalues_valid) != 0
        )
        out["details"]["proj_energy_ratio"] = float(rep.proj_energy_ratio)
        out["details"]["defect_norm"] = float(rep.defect_norm)

        N = 16
        ids = (ct.c_int64 * N)(*range(N))
        omega12 = _zeros(ct.c_int32, N)
        step = _zeros(ct.c_uint64, N)
        for i in range(N):
            step[i] = 10 + i
        last_byte = _zeros(ct.c_uint8, N)
        for i in range(N):
            last_byte[i] = 0xAA
        word4 = (ct.c_uint8 * (4*N))(*(0 for _ in range(4*N)))
        chi_h = (ct.c_uint16 * (64*N))(*(0 for _ in range(64*N)))
        shell_h = (ct.c_uint16 * (7*N))(*(0 for _ in range(7*N)))
        fam_h = (ct.c_uint16 * (4*N))(*(0 for _ in range(4*N)))
        osig = _zeros(ct.c_int32, N)
        pO = _zeros(ct.c_uint16, N)
        pE = _zeros(ct.c_uint16, N)
        pbit = _zeros(ct.c_uint8, N)
        rkey = _zeros(ct.c_uint32, N)
        for c in range(N):
            for j in range(64): chi_h[c*64+j] = (c*100+j)&0xFFFF
            shell_h[c*7+(c%7)] = 3+(c%4); fam_h[c*4+(c%4)] = 11+c
        probe_bufs = CellBuffers(
            omega12,
            step,
            last_byte,
            (ct.c_uint8 * N)(),
            word4,
            (ct.c_uint8 * (64 * N))(),
            (ct.c_uint8 * N)(),
            (ct.c_uint8 * N)(),
            chi_h,
            shell_h,
            (ct.c_uint8 * (64 * N))(),
            fam_h,
            osig,
            pO,
            pE,
            pbit,
            rkey,
        )
        batch = gyrograph_emit_slcp_batch(N, ids, probe_bufs)
        out["details"]["multi_cell_slcp_pass"] = len(batch)==N and len({tuple(b.spectral64) for b in batch})>1
    except Exception as e:
        out["status"] = "fail"; out["error"] = str(e)
    return out


def run_perplexity_comparison(
    *,
    timeout: float = 1800.0,
    n_ctx: int = 512,
    witness_mode: str = "off",
) -> dict[str, Any]:
    cfg = get_gyroscopic_llm_config()
    calib_text = (PERPLEXITY_CALIBRATION_TEXT.strip() + "\n") * max(1, DEFAULT_PERPLEXITY_REPEAT)
    calib_path = DATA_DIR / "_perplexity_calib.txt"
    calib_path.write_text(calib_text, encoding="utf-8")

    results: dict[str, Any] = {}
    for mode in ("stock", "gyroscopic"):
        env = _clean_llama_env(mode, trace=False, witness_mode=witness_mode)

        extra_args = [
            "-c", str(n_ctx),
            "--flash-attn", "off",
        ]

        t0 = time.perf_counter()
        try:
            pr = run_llama_perplexity(
                cfg,
                corpus_path=str(calib_path),
                extra_args=extra_args,
                env=env,
                timeout_sec=timeout,
            )
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            results[mode] = {"error": "timeout", "elapsed": timeout}
            continue
        except FileNotFoundError as e:
            results[mode] = {"error": str(e), "elapsed": 0.0}
            continue

        ppl = pr.get("ppl")

        rc_raw = pr.get("rc", -1)
        rc_val = int(rc_raw) if isinstance(rc_raw, (int, float, str)) else -1
        results[mode] = {
            "perplexity": ppl,
            "elapsed": elapsed,
            "rc": rc_val,
        }

    s_ppl = results.get("stock", {}).get("perplexity")
    g_ppl = results.get("gyroscopic", {}).get("perplexity")
    quality = _perplexity_quality_flags(s_ppl, g_ppl)
    if quality["ppl_delta_pct"] is not None:
        results["ppl_delta_pct"] = quality["ppl_delta_pct"]
    else:
        results["ppl_delta_pct"] = None
    results["quality_equivalent"] = quality["quality_equivalent"]
    results["quality_improved_on_probe"] = quality["quality_improved_on_probe"]
    results["quality_degraded_on_probe"] = quality["quality_degraded_on_probe"]
    results["quality_preserved"] = quality["quality_equivalent"]

    try:
        calib_path.unlink()
    except Exception:
        pass

    return results


def _perplexity_quality_flags(
    stock_ppl: Any,
    gyro_ppl: Any,
    *,
    tolerance_pct: float = 1.0,
) -> dict[str, float | bool | None]:
    out: dict[str, float | bool | None] = {
        "ppl_delta_pct": None,
        "quality_equivalent": None,
        "quality_improved_on_probe": None,
        "quality_degraded_on_probe": None,
    }
    try:
        stock_v = float(stock_ppl)
        gyro_v = float(gyro_ppl)
    except (TypeError, ValueError):
        return out
    if stock_v <= 0.0:
        return out
    delta_pct = (gyro_v - stock_v) * 100.0 / stock_v
    out["ppl_delta_pct"] = delta_pct
    if abs(delta_pct) <= tolerance_pct:
        out["quality_equivalent"] = True
    elif delta_pct > 0.0:
        out["quality_degraded_on_probe"] = True
    else:
        out["quality_improved_on_probe"] = True
    return out


def print_full_report(
    results: list[RunResult],
    by_prompt: dict[int, dict[str, RunResult]],
    climate: dict[str, Any],
    kv_pri: dict[str, Any],
    multi_cell: dict[str, Any],
    ppl: dict[str, Any] | None = None,
) -> None:
    """Single stdout report: llama runs, native probes, hybrid trace (no extra log files)."""
    w = 65
    print("\n" + "=" * w)
    print("GYROSCOPIC BENCHMARK REPORT")
    print("=" * w)

    print("\n--- llama-cli runs (stock vs gyroscopic) ---\n")
    print(f"{'Mode':<10} {'Prompt':<35} {'Time':>6} {'TPS':>6} {'Status':<8}")
    print("-" * w)
    pair_speed_ratios: list[float] = []
    skipped_pairs = 0
    failed_stock_pairs = 0
    for r in results:
        status = _format_run_status(r)
        if r.gen_tps is not None:
            tps_str = f"{r.gen_tps:.1f}"
        elif r.timed_out or r.silent_kill:
            tps_str = "--"
        else:
            tps_str = "n/a"
        print(f"{r.mode:<10} {r.prompt[:33]:<35} {r.elapsed:>5.1f}s {tps_str:>6} {status:<8}")
    for prompt_idx in sorted(by_prompt.keys()):
        pair = by_prompt[prompt_idx]
        s = pair.get("stock")
        g = pair.get("gyroscopic")
        if s is None or g is None:
            skipped_pairs += 1
            continue
        if not s.ok:
            failed_stock_pairs += 1
            continue
        if not g.ok:
            skipped_pairs += 1
            continue
        if s.gen_tps is None or g.gen_tps is None:
            skipped_pairs += 1
            continue
        pair_speed_ratios.append(g.gen_tps / s.gen_tps)

    print("-" * w)
    if pair_speed_ratios:
        ratio = statistics.mean(pair_speed_ratios)
        print(f"Throughput Ratio (Gyro/Stock): {ratio:.3f}x")
        if failed_stock_pairs:
            print(f"Pairs skipped from ratio due failed stock baseline: {failed_stock_pairs}")
        if skipped_pairs:
            print(f"Pairs skipped from ratio due invalid/missing runs: {skipped_pairs}")
        print(f"Valid comparison pairs: {len(pair_speed_ratios)}")
        print("\n--- wall time + output parity by prompt ---")
        for prompt_idx in sorted(by_prompt.keys()):
            pair = by_prompt[prompt_idx]
            s = pair.get("stock")
            g = pair.get("gyroscopic")
            if s is None or g is None:
                continue
            wall_ratio = g.elapsed / s.elapsed if s.elapsed > 0.0 else None
            if s.ok and g.ok:
                output_hash_match = g.stdout_hash == s.stdout_hash if ENABLE_HASH_MATCH else None
            else:
                output_hash_match = None
            wall_ratio_text = f"{wall_ratio:.2f}x" if wall_ratio is not None else "--"
            print(
                f"{prompt_idx + 1:<4} {wall_ratio_text:<8} "
                f"hash_match={'disabled' if not ENABLE_HASH_MATCH else str(output_hash_match).lower():<5} "
                f"{s.prompt[:30]}"
            )
    else:
        print("Throughput Ratio (Gyro/Stock): n/a (no completed run with parsed TPS)")

    g = next((r.trace for r in reversed(results) if r.mode == "gyroscopic"), None)
    if g and g.structured_rows + g.dense_rows > 0:
        total = g.structured_rows + g.dense_rows
        print(f"Hybrid Routing: {g.structured_rows}/{total} structured ({g.structured_rows/total*100:.1f}%)")
        if g.pq_rows > 0 or g.dq_rows > 0:
            print(f"P_Q rows: {g.pq_rows} | D_Q rows: {g.dq_rows}")
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
        native_rows = int(g_load.structured_rows)
        spectral_sparse_rows = int(g_load.spectral_sparse_rows)
        stock_rows = int(g_load.dense_rows) - spectral_sparse_rows
        if stock_rows < 0:
            stock_rows = 0
        print("\n--- benchmark routing + requested metrics ---\n")
        _print_section(
            "execution_route row counts",
            [
                ("STOCK", stock_rows),
                ("SPECTRAL_SPARSE", spectral_sparse_rows),
                ("NATIVE", native_rows),
            ],
        )
        _print_section(
            "requested benchmark rows and blocks",
            [
                ("pq_chi_rows", g_load.pq_chi_rows),
                ("pq_shell_rows", g_load.pq_shell_rows),
                ("dq_k4_rows", g_load.dq_k4_rows),
                ("dq_q8_rows", g_load.dq_q8_rows),
                ("structured_rows", g_load.structured_rows),
                ("dense_rows", g_load.dense_rows),
                ("generic_blocks", g_load.generic_blocks),
            ],
        )

    g_reg = next((r.trace for r in reversed(results) if r.mode == "gyroscopic"), None)
    if g_reg is not None:
        g_load = g_reg
        print("\n--- gyro tensor registration (stderr GYRO_REG) ---\n")
        if g_reg.reg_load_tensors_scanned > 0 or g_reg.reg_load_registry_entries > 0:
            print(f"  tensors_scanned: {g_reg.reg_load_tensors_scanned}")
            print(f"  q8_0_tensors: {g_reg.reg_load_q8_0_tensors}")
            print(f"  registry_entries: {g_reg.reg_load_registry_entries}")
            print(f"  {'index-built-slots':<30} {g_reg.index_built_slots:>8}")
            print(f"  {'index-total-entries':<30} {g_reg.index_total_entries:>8}")
            print(f"  {'index-query-hits':<30} {g_reg.index_query_hits:>8}")
            print(f"  {'index-query-misses':<30} {g_reg.index_query_misses:>8}")
        else:
            print("  (no GYRO_REG line parsed; gyroscopic tensor registration did not run or stderr capture missed it)")
        if g_reg.reg_class_counts is not None:
            cc = g_reg.reg_class_counts
            total = sum(cc)
            _print_distribution(
                "exact membership distribution (all registered blocks)",
                _CLASS_DISPLAY_NAMES,
                cc,
            )
            structured = total - cc[4] if len(cc) > 4 else 0
            pct_structured = 100.0 * structured / total if total > 0 else 0.0
            print(f"  {'structured':<16} {structured:>8}  ({pct_structured:.1f}%)")
            print(f"  {'total':<16} {total:>8}")
        if g_reg.reg_selected_projection_counts is not None:
            cc = g_reg.reg_selected_projection_counts
            total = sum(cc)
            _print_distribution(
                "projection class distribution (all registered blocks)",
                _CLASS_DISPLAY_NAMES,
                cc,
            )
            structured = total - cc[4] if len(cc) > 4 else 0
            pct_structured = 100.0 * structured / total if total > 0 else 0.0
            print(f"  {'structured':<16} {structured:>8}  ({pct_structured:.1f}%)")
            print(f"  {'total':<16} {total:>8}")
        if g_reg.reg_residual_format_counts is not None:
            cc = g_reg.reg_residual_format_counts
            _print_distribution(
                "D_Q defect format distribution (all registered blocks)",
                ["dq-none", "dq-k4-packed", "dq-q8-exact", "dq-projectedbackfill-debug"],
                cc,
            )
        if (
            g_reg.shell_radial_blocks
            or g_reg.shell_gauge_blocks
            or g_reg.chi_invariant_blocks
            or g_reg.chi_gauge_blocks
            or g_reg.generic_blocks
        ):
            rows_by_class = [
                (name, value) for name, value in zip(
                    _CLASS_DISPLAY_NAMES,
                    [g_reg.shell_radial_rows, g_reg.shell_gauge_rows, g_reg.chi_invariant_rows, g_reg.chi_gauge_rows, g_reg.generic_rows],
                )
            ]
            blocks_by_class = [
                (name, value) for name, value in zip(
                    _CLASS_DISPLAY_NAMES,
                    [g_reg.shell_radial_blocks, g_reg.shell_gauge_blocks, g_reg.chi_invariant_blocks, g_reg.chi_gauge_blocks, g_reg.generic_blocks],
                )
            ]
            print("\n--- hot-path class realization (trace footer) ---\n")
            print("  rows:")
            for name, v in rows_by_class:
                print(f"    {name:<14} {v:>8}")
            print("  blocks:")
            for name, v in blocks_by_class:
                print(f"    {name:<14} {v:>8}")
        if (
            g_load.selected_shell_radial_blocks
            or g_load.selected_shell_gauge_blocks
            or g_load.selected_chi_invariant_blocks
            or g_load.selected_chi_gauge_blocks
            or g_load.selected_generic_blocks
            or g_load.projection_shell_radial_blocks
            or g_load.projection_shell_gauge_blocks
            or g_load.projection_chi_invariant_blocks
            or g_load.projection_chi_gauge_blocks
            or g_load.projection_generic_blocks
        ):
            blocks_by_class = [
                ("shell-radial", g_load.selected_shell_radial_blocks),
                ("shell-x-gauge", g_load.selected_shell_gauge_blocks),
                ("chi-invariant", g_load.selected_chi_invariant_blocks),
                ("chi-x-gauge", g_load.selected_chi_gauge_blocks),
                ("generic", g_load.selected_generic_blocks),
            ]
            _print_section("projection pressure (hot-path blocks)", blocks_by_class)
        if (
            g_load.exact_containment_shell_radial_blocks
            or g_load.exact_containment_shell_gauge_blocks
            or g_load.exact_containment_chi_invariant_blocks
            or g_load.exact_containment_chi_gauge_blocks
            or g_load.exact_containment_generic_blocks
            or g_load.certified_shell_radial_blocks
            or g_load.certified_shell_gauge_blocks
            or g_load.certified_chi_invariant_blocks
            or g_load.certified_chi_gauge_blocks
            or g_load.certified_generic_blocks
        ):
            blocks_by_class = [
                ("shell-radial", g_load.exact_containment_shell_radial_blocks),
                ("shell-x-gauge", g_load.exact_containment_shell_gauge_blocks),
                ("chi-invariant", g_load.exact_containment_chi_invariant_blocks),
                ("chi-x-gauge", g_load.exact_containment_chi_gauge_blocks),
                ("generic", g_load.exact_containment_generic_blocks),
            ]
            _print_section("exact containment pressure (hot-path blocks)", blocks_by_class)
        if (
            g_load.exec_shell_radial_blocks
            or g_load.exec_shell_gauge_blocks
            or g_load.exec_chi_invariant_blocks
            or g_load.exec_chi_gauge_blocks
            or g_load.exec_generic_blocks
        ):
            blocks_by_class = [
                ("shell-radial", g_load.exec_shell_radial_blocks),
                ("shell-x-gauge", g_load.exec_shell_gauge_blocks),
                ("chi-invariant", g_load.exec_chi_invariant_blocks),
                ("chi-x-gauge", g_load.exec_chi_gauge_blocks),
                ("generic", g_load.exec_generic_blocks),
            ]
            _print_section("kernel-class pressure (hot-path blocks)", blocks_by_class)
        if (
            g_load.dq_none_blocks
            or g_load.dq_k4_blocks
            or g_load.dq_q8_exact_blocks
            or g_load.dq_backfill_debug_blocks
            or g_load.residual_none_blocks
            or g_load.residual_k4_blocks
            or g_load.residual_exact_q8_blocks
            or g_load.residual_backfill_debug_blocks
        ):
            residual_blocks = [
                ("none", g_load.dq_none_blocks or g_load.residual_none_blocks),
                ("k4-packed", g_load.dq_k4_blocks or g_load.residual_k4_blocks),
                ("exact-q8", g_load.dq_q8_exact_blocks or g_load.residual_exact_q8_blocks),
                (
                    "projected-backfill-debug",
                    g_load.dq_backfill_debug_blocks
                    or g_load.residual_backfill_debug_blocks,
                ),
            ]
            _print_section("D_Q defect pressure (hot-path blocks)", residual_blocks)
        if (
            g_load.compiled_k4_residual_blocks
            or g_load.compiled_k4_residual_direct_decode_calls
            or g_load.exact_q8_generic_blocks
            or g_load.chi_gauge_dq_kernel_calls
            or g_load.chi_gauge_backfill_calls
        ):
            _print_section(
                "native D_Q execution (hot-path)",
                [
                    ("compiled-k4-residual-blocks", g_load.compiled_k4_residual_blocks),
                    (
                        "compiled-k4-residual-direct-decode-calls",
                        g_load.compiled_k4_residual_direct_decode_calls,
                    ),
                    ("exact-q8-generic-blocks", g_load.exact_q8_generic_blocks),
                    ("pq_chi_gauge__dq_k4", g_load.exec_pq_chi_gauge_dq_k4_blocks),
                    ("pq_chi_gauge__dq_q8", g_load.exec_pq_chi_gauge_dq_q8_blocks),
                    ("pq_generic__dq_q8", g_load.exec_pq_generic_dq_q8_blocks),
                    ("chi-gauge-dq-kernel-calls", g_load.chi_gauge_dq_kernel_calls),
                    ("chi-gauge-backfill-calls", g_load.chi_gauge_backfill_calls),
                ],
            )
        if (
            g_load.kernel_chi_gauge_k4_blocks
            or g_load.kernel_chi_gauge_exact_q8_blocks
            or g_load.kernel_generic_exact_q8_blocks
            or g_load.kernel_generic_k4_blocks
            or g_load.kernel_projected_backfill_blocks
        ):
            _print_section(
                "native kernel family (hot-path blocks)",
                [
                    ("pq_chi_gauge__dq_k4", g_load.kernel_chi_gauge_k4_blocks),
                    ("pq_chi_gauge__dq_q8", g_load.kernel_chi_gauge_exact_q8_blocks),
                    ("pq_generic__dq_q8", g_load.kernel_generic_exact_q8_blocks),
                    ("pq_generic__dq_k4", g_load.kernel_generic_k4_blocks),
                    ("projected-backfill", g_load.kernel_projected_backfill_blocks),
                ],
            )
        if g_load.decode_events:
            _print_section(
                "decode policy realization (hot-path)",
                [
                    ("events", g_load.decode_events),
                    ("grouped-dispatch", g_load.decode_grouped_dispatch_events),
                    ("ungrouped-dispatch", g_load.decode_ungrouped_dispatch_events),
                    ("kv-evict-events", g_load.decode_kv_evict_events),
                    ("mean-chi-distance", g_load.decode_mean_chi_distance),
                    ("max-chi-distance", g_load.decode_max_chi_distance_seen),
                    ("mean-kv-priority", g_load.decode_mean_kv_priority),
                ],
            )
        if g_load.chi_gauge_cache_hits or g_load.chi_gauge_cache_misses:
            _print_section(
                "chi-gauge cache realization",
                [
                    ("cache-hits", g_load.chi_gauge_cache_hits),
                    ("cache-misses", g_load.chi_gauge_cache_misses),
                    ("cache-stores", g_load.chi_gauge_cache_stores),
                    ("cache-hit-rate", g_load.chi_gauge_cache_hit_rate),
                ],
            )

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
            "decode_group_count",
            "decode_group_sizes",
            "kv_priority_min",
            "kv_priority_mean",
            "kv_priority_max",
            "m2_empirical_mean",
            "m2_empirical_stdev",
            "steps_ok",
        ):
            if k in multi_cell:
                print(f"  {k}: {multi_cell[k]}")
        if multi_cell.get("error") and not multi_cell.get("skipped"):
            print(f"  error: {multi_cell['error']}")

    if ppl and not ppl.get("skipped"):
        print("\n--- perplexity comparison (language quality) ---\n")
        for mode in ("stock", "gyroscopic"):
            mr = ppl.get(mode, {})
            p = mr.get("perplexity")
            e = mr.get("elapsed", 0)
            if p is not None:
                print(f"  {mode}: PPL = {p:.4f} ({e:.1f}s)")
            elif mr.get("error"):
                print(f"  {mode}: {mr['error']}")
        delta = ppl.get("ppl_delta_pct")
        eq = ppl.get("quality_equivalent")
        improved = ppl.get("quality_improved_on_probe")
        degraded = ppl.get("quality_degraded_on_probe")
        if delta is not None:
            print(f"  PPL delta: {delta:+.4f}%")
        print(f"  Quality equivalent (within 1.0%): {eq}")
        print(f"  Quality improved on probe: {improved}")
        print(f"  Quality degraded on probe: {degraded}")

    print("\n" + "=" * w)


def _resolve_bench_mode(args: Any) -> str:
    """
    Resolve benchmark mode with legacy flag compatibility:
    compare, stock-only, verify, interop, gyro-only.
    """
    if getattr(args, "mode", None):
        return str(getattr(args, "mode"))
    if getattr(args, "interop_modes", None) is not None:
        return "interop"
    if getattr(args, "verify", False):
        return "verify"
    if getattr(args, "no_gyro", False):
        return "stock-only"
    return "compare"


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Default: stock+gyroscopic smoke run (one prompt), no warmup, "
            "native climate + multi-cell probes; "
            f"writes only {OUT_JSON}."
        )
    )
    p.add_argument(
        "--mode",
        default=None,
        choices=("compare", "stock-only", "verify", "interop", "gyro-only"),
        help="Benchmark mode: stock+gyro (compare), stock-only, verify, interop, or gyro-only.",
    )
    p.add_argument("--no-gyro", action="store_true", help="Run stock only (skip gyroscopic). Legacy alias for --mode stock-only.")
    p.add_argument(
        "--verify",
        action="store_true",
        help="Enable sampled parity witness. Legacy alias for --mode verify.",
    )
    p.add_argument(
        "--perplexity",
        action="store_true",
        help="Enable perplexity comparison (default: disabled).",
    )
    p.add_argument("--timeout", type=float, default=None, help="Override TIMEOUT_STOCK and TIMEOUT_GYRO.")
    p.add_argument("--n-predict", type=int, default=None, metavar="N", help=f"Tokens to generate (default {DEFAULT_N_PREDICT}).")
    p.add_argument(
        "--warmup",
        action="store_true",
        help="Run a short warmup before measured runs, based on selected mode.",
    )
    p.add_argument(
        "--prompt-suite",
        default="smoke",
        choices=tuple(sorted(PROMPT_SUITES.keys())),
        help="Named prompt suite for the benchmark ladder.",
    )
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable optional gyroscopic trace telemetry parsing.",
    )
    p.add_argument("--skip-build", action="store_true", help="Use existing llama-cli binary without rebuild.")
    p.add_argument(
        "--force-build",
        action="store_true",
        help="Force rebuilding llama.cpp even if a previous binary exists.",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="Build native surfaces and exit before prompt runs.",
    )
    p.add_argument(
        "--interop-modes",
        default=None,
        help="Comma-separated interop modes for sweep (default sweep off).",
    )
    p.add_argument(
        "--interop-decode-max-batch",
        type=int,
        default=None,
        help="Optional decode max batch override while sweeping.",
    )
    p.add_argument(
        "--interop-decode-max-chi-distance",
        type=int,
        default=None,
        help="Optional chi-distance override while sweeping.",
    )
    p.add_argument(
        "--interop-decode-max-shell-delta",
        type=int,
        default=None,
        help="Optional shell-delta override while sweeping.",
    )
    p.add_argument(
        "--runtime-analyze",
        action="store_true",
        help=(
            "Enable additional runtime-analysis environment flags for gyroscopic diagnostics. "
            "This must not change gyroscopic ownership semantics."
        ),
    )
    p.add_argument(
        "--show-progress-lines",
        action="store_true",
        help=(
            "Show per-run progress print lines during execution. "
            "The final report remains unchanged."
        ),
    )
    args = p.parse_args()

    if args.skip_build:
        os.environ["GYROLABE_SKIP_LLAMA_BUILD"] = "1"
        print("[bench] Using existing llama-cli binary (--skip-build).")
    else:
        if args.force_build:
            os.environ.pop("GYROLABE_SKIP_LLAMA_BUILD", None)
        print("[bench] Building native surfaces if needed...")
    try:
        build_llama_cpp_if_needed()
    except RuntimeError as exc:
        print(f"[bench] native build failed: {exc}")
        return 1

    if args.build_only:
        print("[bench] build-only mode complete.")
        return 0

    prompts = list(PROMPT_SUITES[args.prompt_suite])
    timeout_stock = float(args.timeout) if args.timeout is not None else TIMEOUT_STOCK
    timeout_gyro = float(args.timeout) if args.timeout is not None else TIMEOUT_GYRO
    n_predict = int(args.n_predict) if args.n_predict is not None else DEFAULT_N_PREDICT
    trace_enabled = not args.no_trace
    print_progress = args.show_progress_lines
    results: list[RunResult] = []
    if args.interop_modes is not None:
        if args.mode is None:
            mode = "interop"
        else:
            mode = _resolve_bench_mode(args)
    else:
        mode = _resolve_bench_mode(args)
    if mode == "interop":
        sweep_modes = _parse_interop_mode_list(args.interop_modes or "")
        if not sweep_modes:
            sweep_modes = list(_INTEROP_SWEEP_DEFAULT_MODES)
    else:
        sweep_modes = [None]
        if args.interop_modes is not None:
            print("[bench] --interop-modes ignored unless --mode interop.")

    run_stock = mode in {"compare", "verify", "interop", "stock-only"}
    run_gyro = mode in {"compare", "verify", "interop", "gyro-only"}
    witness_mode = "sampled" if mode == "verify" else "off"
    bench_mode = mode
    mode_label = "stock + gyroscopic" if run_stock and run_gyro else ("gyroscopic only" if run_gyro else "stock only")
    print(
        f"[bench] {mode_label} | {len(prompts)} prompt(s) | n_predict={n_predict} | "
        f"timeouts stock={timeout_stock:.0f}s gyro={timeout_gyro:.0f}s | "
        f"silent_kill={SILENT_KILL_SEC:.0f}s | warmup={'on' if args.warmup else 'off'}"
        f" | bench_mode={bench_mode}"
        f"{' | interop=' + ','.join(m for m in sweep_modes if m) if sweep_modes != [None] else ''}"
        f"{' | trace=on' if trace_enabled else ' | trace=off'}"
    )

    if args.warmup and prompts:
        pr0 = prompts[0]
        w_n = max(1, min(8, n_predict))
        w_to = min(300.0, timeout_stock, timeout_gyro)
        if run_stock:
            print(f"[bench] Warmup stock ({w_n} tok, timeout {w_to:.0f}s)...")
            _ = run_llama_streaming("stock", pr0, w_n, w_to, -1, trace=False)
        if run_gyro:
            print(f"[bench] Warmup gyroscopic ({w_n} tok, timeout {w_to:.0f}s)...")
            _ = run_llama_streaming(
                "gyroscopic",
                pr0,
                w_n,
                w_to,
                -1,
                trace=trace_enabled,
                witness_mode=witness_mode,
            )
        time.sleep(0.05)

    for i, pr in enumerate(prompts):
        r_s = None
        if run_stock:
            r_s = run_llama_streaming("stock", pr, n_predict, timeout_stock, i, trace=False)
            results.append(r_s)
            if print_progress:
                tps_s = f"{r_s.gen_tps:.1f}" if r_s.gen_tps is not None else "--"
                print(f"  Stock {i+1}/{len(prompts)}: {r_s.elapsed:.1f}s | TPS:{tps_s} | {r_s.stdout_hash}")
                if r_s.silent_kill:
                    print("  [diag] Stock run had no output for a while; check system health.")
                elif r_s.timed_out:
                    print("  [diag] Stock run timed out.")

        if run_gyro:
            for interop_mode in sweep_modes:
                decode_batch_size = args.interop_decode_max_batch
                decode_chi_distance = args.interop_decode_max_chi_distance
                decode_shell_delta = args.interop_decode_max_shell_delta
                if interop_mode in {"advisory", "approximate_derived"}:
                    if decode_batch_size is None:
                        decode_batch_size = 2
                    if decode_chi_distance is None:
                        decode_chi_distance = 0
                    if decode_shell_delta is None:
                        decode_shell_delta = 0
                extra_env = _build_interop_env(
                    interop_mode,
                    decode_max_batch_size=decode_batch_size,
                    decode_max_chi_distance=decode_chi_distance,
                    decode_max_shell_delta=decode_shell_delta,
                )
                if args.runtime_analyze:
                    extra_env["GGML_GYROSCOPIC_RUNTIME_ANALYZE"] = "1"
                run_mode_label = f"{interop_mode}" if interop_mode else "default"
                r_g = run_llama_streaming(
                    "gyroscopic",
                    pr,
                    n_predict,
                    timeout_gyro,
                    i,
                    trace=trace_enabled,
                    witness_mode=witness_mode,
                    interop_mode=interop_mode,
                    extra_env=extra_env,
                )
                results.append(r_g)
                status = _format_run_status(r_g)
                tps_g = f"{r_g.gen_tps:.1f}" if r_g.gen_tps is not None else "--"
                if r_s is not None:
                    wall_ratio = r_g.elapsed / r_s.elapsed if r_s.elapsed > 0.0 else None
                    wall_ratio_str = f"{wall_ratio:.2f}x" if wall_ratio is not None else "--"
                    if r_g.ok and r_s.ok:
                        output_hash_match = r_g.stdout_hash == r_s.stdout_hash if ENABLE_HASH_MATCH else None
                    else:
                        output_hash_match = None
                else:
                    wall_ratio_str = "--"
                    output_hash_match = None
                if print_progress:
                    print(
                        f"  Gyro({run_mode_label}) {i+1}/{len(prompts)}: {r_g.elapsed:.1f}s | "
                        f"TPS:{tps_g} | wall/stock={wall_ratio_str} | "
                        f"hash_match={'disabled' if not ENABLE_HASH_MATCH else str(output_hash_match).lower()} | {status} | {r_g.stdout_hash}"
                    )
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
                if r_s is None:
                    continue
                if ENABLE_HASH_MATCH and r_g.stdout_hash != r_s.stdout_hash:
                    if not r_s.ok or not r_g.ok:
                        if print_progress:
                            print("  [bench] Parity: skipped for timing comparison (one side failed).")
                    else:
                        if print_progress:
                            print(
                                "  [bench] Parity: stdout hash differs "
                                f"(stock={r_s.stdout_hash} gyro={r_g.stdout_hash})"
                            )

    if bench_mode == "stock-only":
        print("[bench] Gyroscopic passes skipped (--stock-only).")

    try:
        print("[bench] Post-run: climate probe...")
        climate = run_qubec_climate_probe()
    except BaseException as exc:
        climate = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[bench] Post-run climate probe failed: {climate['error']}")

    try:
        print("[bench] Post-run: KV priority probe...")
        kv_pri = run_kv_priority_illustrative()
    except BaseException as exc:
        kv_pri = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[bench] Post-run KV priority probe failed: {kv_pri['error']}")

    try:
        print("[bench] Post-run: multi-cell native probe...")
        multi_cell = run_multi_cell_word4_benchmark(MULTI_CELL_NATIVE_LABELS)
    except BaseException as exc:
        multi_cell = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[bench] Post-run multi-cell probe failed: {multi_cell['error']}")
    requested_perplexity = args.perplexity
    run_perplexity = bool(requested_perplexity and run_gyro)

    if run_perplexity:
        print("[bench] Running perplexity comparison (stock vs gyroscopic)...")
        ppl_result: dict[str, Any] = run_perplexity_comparison(
            timeout=timeout_gyro * 2.0,
            witness_mode="off",
        )
    elif bench_mode == "stock-only":
        ppl_result = {"skipped": True, "reason": "stock-only mode"}
    elif not requested_perplexity:
        ppl_result = {"skipped": True, "reason": "disabled by default"}

    by_prompt: dict[int, dict[str, RunResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_idx, {})[r.mode] = r
    pair_metrics: list[dict[str, Any]] = []
    for prompt_idx, modes in by_prompt.items():
        s = modes.get("stock")
        g = modes.get("gyroscopic")
        if s is None or g is None:
            continue
        stock_ok = s.ok
        gyro_ok = g.ok
        pair_metrics.append(
            {
                "prompt_idx": prompt_idx,
                "prompt": s.prompt,
                "stock_elapsed": s.elapsed,
                "gyro_elapsed": g.elapsed,
                "stock_rc": s.rc,
                "gyro_rc": g.rc,
                "stock_timed_out": s.timed_out,
                "gyro_timed_out": g.timed_out,
                "stock_silent_kill": s.silent_kill,
                "gyro_silent_kill": g.silent_kill,
                "stock_status": s.status_label,
                "gyro_status": g.status_label,
                "wall_time_ratio": (
                    g.elapsed / s.elapsed
                    if stock_ok and gyro_ok and s.elapsed > 0.0 else None
                ),
                "output_hash_match": (
                    (g.stdout_hash == s.stdout_hash) if (stock_ok and gyro_ok) else None
                ),
                "stock_wall_effective_gen_tps": s.wall_effective_gen_tps,
                "gyro_wall_effective_gen_tps": g.wall_effective_gen_tps,
                "wall_effective_gen_tps_ratio": (
                    (g.wall_effective_gen_tps / s.wall_effective_gen_tps)
                    if (
                        stock_ok
                        and gyro_ok
                        and s.wall_effective_gen_tps is not None
                        and g.wall_effective_gen_tps is not None
                        and s.wall_effective_gen_tps > 0.0
                    )
                    else None
                ),
                "stock_llama_gen_eval_ms": s.llama_gen_eval_ms,
                "gyro_llama_gen_eval_ms": g.llama_gen_eval_ms,
                "stock_measurement_exposure": _compact_measurement_exposure(s),
                "gyro_measurement_exposure": _compact_measurement_exposure(g),
            }
        )

    print("[bench] Post-run: report + JSON export...")
    try:
        print_full_report(results, by_prompt, climate, kv_pri, multi_cell, ppl_result)
    except BaseException as exc:
        print(f"[bench] Report generation failed: {type(exc).__name__}: {exc}")
    runs_export: list[dict[str, Any]] = []
    for r in results:
        row = asdict(r)
        row["trace"] = _compact_trace(r)
        row["measurement_exposure"] = _compact_measurement_exposure(r)
        runs_export.append(row)
    payload: dict[str, Any] = {
        "scope": BENCH_SCOPE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bench_mode": bench_mode,
        "witness_mode": witness_mode if run_gyro else "off",
        "prompt_suite": args.prompt_suite,
        "measurement_legend": MEASUREMENT_LEGEND,
        "runs": runs_export,
        "pair_metrics": pair_metrics,
        "climate_probe": climate,
        "kv_priority_illustrative": kv_pri,
        "multi_cell_word4": multi_cell,
        "perplexity": ppl_result,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] Exported: {OUT_JSON}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
