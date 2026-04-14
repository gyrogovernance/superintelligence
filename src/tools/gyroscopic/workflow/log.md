# Gyroscopic integration log

Current status: active llama.cpp execution path is QuBEC lowering with registry-guided
structured matmul plus GyroGraph climate/orchestration. Legacy packed K4 arithmetic is not part of the live surface.

## Canonical paths and responsibilities

Paths below are relative to the repo root unless prefixed with `external/llama.cpp/`.

| Path | Role |
|------|------|
| `external/llama.cpp/ggml/include/ggml-gyroscopic-graph.h` | Graph API: token feed, per-seq M2, resonance key (when `GGML_USE_GYROSCOPIC_GRAPH`). |
| `external/llama.cpp/ggml/src/ggml-gyroscopic/gyroscopic-backend.h` | Bridge API: gyroscopic mode/strict/trace, `ggml_gyroscopic_can_use`, `ggml_gyroscopic_mul_mat`, q8dot override counters. |
| `external/llama.cpp/ggml/src/ggml-gyroscopic/gyroscopic-backend.cpp` | Main bridge: validates geometry, dispatches `gyrolabe_qubec_matmul_q8_0`, merges row/dispatch stats, stderr summaries (`GyroMatMul`, `GyroRows`, `GyroDispatch`, `GyroResidual`, `GyroGraph`). |
| `external/llama.cpp/ggml/src/ggml-gyroscopic/ggml-cpu.c` | `MUL_MAT` hooks: chunk and fast Q8_0 paths call `ggml_gyroscopic_mul_mat` when eligible. |
| `external/llama.cpp/ggml/src/ggml-gyroscopic/vec.cpp` | F32 vec-dot hook; gyroscopic path returns `false` (no f32 override). |
| `external/llama.cpp/ggml/src/ggml-gyroscopic/gyroscopic-base.c` | Per-cell GyroGraph buffers in ggml-base; `ggml_gyroscopic_graph_feed_token` and M2 stats (init guarded by `ggml_critical_section_*`). |
| `src/tools/gyroscopic/gyrograph_policy.c` | Loads `GGML_GYROSCOPIC`, `GGML_GYROSCOPIC_STRICT`, `GGML_GYROSCOPIC_TRACE` once per process into `gyro_policy`. |
| `src/tools/gyroscopic/gyrograph.c` | Omega stepping, rolling chi/shell/family memories, SLCP emission, native batch ingest. |
| `src/tools/gyroscopic/gyrograph.h` | GyroGraph C API used by `gyroscopic-base.c` and backend. |
| `src/tools/gyroscopic/gyrolabe_registry.c` | Registers Q8_0 weight tensors, tiles into 64x64 blocks, builds structured spectra and optional residual blocks for matmul. |
| `src/tools/gyroscopic/gyrolabe_wht.c` | In-place WHT64 (`gyrolabe_wht64_f32_inplace`), used by registry and QuBEC matmul. |
| `src/tools/gyroscopic/gyrolabe_transforms.c` | Krawtchouk7, K4Char4, K4 lattice helpers (Python `gyrolabe_native` and tooling; ggml CPU link set may omit if unused by that build). |
| `src/tools/gyroscopic/gyrolabe_qubec_matmul.c` | QuBEC Q8_0 matmul: registry-guided structured path plus dense Q8_0 fallback, strict-mode witness checks, call statistics. |
| `src/tools/gyroscopic/ops_build.py` | Builds `gyrolabe_native` and optionally llama.cpp with `GGML_GYROSCOPIC` CPU backend wired to GyroLabe + GyroGraph sources. |
| `scripts/bench_gyroscopic_llama.py` | Stock vs gyroscopic benchmark; parses stderr trace (`GyroMatMul`, `GyroRows`, `GyroDispatch`, `GyroGraph`). |

## Core math flow (structured external path)

1. **Weights**

   - Q8_0 tensors are registered at runtime (`gyrolabe_registry_register_q8_buffer` / tensor registration) with row stride from ggml.
   - The registry dequantizes each 64x64 tile and fills `gyrolabe_block_info_t` with quotient class, structured eigenvalues (chi x gauge spectral layout), optional `residual_block` and `residual_norm` when the structured reconstruction does not match the tile.
   - Metadata is in-process only (not persisted on disk).

2. **Activations and output blocks**

   - In `gyrolabe_qubec_matmul_q8_0`, each 64-wide weight tile uses registry info when present:
     - Structured path applies the spectral chi x gauge pipeline and adds residual tile contribution when `residual_block` is non-null.
     - In strict mode (`GGML_GYROSCOPIC_STRICT=1`), witness rows compare structured output to dense Q8_0 accumulation; failures increment parity counters and may force dense for that work.
     - If no structured metadata or the kernel rejects the panel, dense Q8_0 dot accumulation is used (`gyromatmul_vec_dot_q8_0_q8_0` AVX2 or reference).
   - `gyrolabe_qubec_call_stats` exposes `structured_attempt_rows`, `structured_rows`, `dense_rows`, `exact_witness_rows`, `parity_mismatch_rows`, `max_abs_row_error`, and class flags `used_radial`, `used_chi`, `used_chi_gauge`.

3. **Exactness baseline**

   - Dense baseline is ggml-style Q8_0 dot product (same family as `ggml_vec_dot_q8_0_q8_0`).
   - Strict mode relies on witness comparison and `parity_mismatch_rows` / `max_abs_row_error` for confidence.

## Runtime policy

Parsed once in `gyrograph_policy.c` via `gyro_policy_get()`:

- **Mode** from `GGML_GYROSCOPIC` (stock vs gyroscopic backend active).
- **Strict** from `GGML_GYROSCOPIC_STRICT`: structured vs dense witness validation and bookkeeping inside QuBEC matmul.
- **Trace** from `GGML_GYROSCOPIC_TRACE`: stderr bind/dims (first hit) and shutdown summary lines.
- **Optional** `GGML_GYROSCOPIC_K4`: described in some integration drafts as a structured-path kill-switch; the `gyroscopic-backend.cpp` in this tree does not read it. If you add it, document it here and in bench scripts.
- **`GGML_GYROSCOPIC_PURE`**: if set to a truthy value (`1`, `y`, …; disabled by `0`, `n`, `f`), process exit runs a postcondition: gyroscopic mode must have credited **`structured_rows > 0`** in the backend accumulators; otherwise stderr prints an error and **`abort()`**. Use this to fail fast when a run used only dense QuBEC rows. Unset by default.

## Hook order in `ggml_compute_forward_mul_mat`

1. Call sites check `ggml_gyroscopic_active()` and layout preconditions, then `ggml_gyroscopic_mul_mat(...)`.
2. `ggml_gyroscopic_mul_mat` applies the same geometry checks as `ggml_gyroscopic_can_use` (preflight without side effects) before calling `gyrolabe_qubec_matmul_q8_0`.
3. The tensor passed for registry / block lookup is always the **Q8_0 weights** (`src1`): `ggml-cpu.c` passes `src1`; the bridge uses `w_tensor` if non-null, else `src1` (never `src0` activations).
4. If the bridge returns false, ggml continues on stock Q8 paths.

## Trace contract (active)

Startup (gyroscopic mode on):

- `GyroMatMul: mode=gyroscopic active=<0|1> trace=<0|1> strict=<0|1> residual=none`  
  (`residual` is reserved for future `gyrograph` residual policy; currently literal `none` in `gyroscopic-backend.cpp`.)

Shutdown (`GGML_GYROSCOPIC_TRACE=1`):

- `GyroMatMul stats: qubec_calls=... radial_calls=... chi_calls=... chi_gauge_calls=... dense_calls=... q8dot_override_calls=...`
- `GyroRows: structured_rows=... dense_rows=... structured_attempt_rows=... exact_witness_rows=... parity_mismatch_rows=... max_abs_row_error=...`
- `GyroDispatch: attempts=... no_structured_fallback=0 kernel_error_fallback=0 scanned_blocks=... no_k64_blocks=... dispatch_entries=<registry entry count>`  
  (The two zero placeholders are reserved for extended dispatch accounting if wired later.)
- `GyroResidual: mode=none`
- `GyroGraph: m2_min=... m2_max=... m2_mean=... cells=...`

## Benchmark (`scripts/bench_gyroscopic_llama.py`)

- Compares normalized stdout between stock and gyroscopic runs.
- Typical parity-oriented pass uses `GGML_GYROSCOPIC_STRICT=1`.
- Parses stderr aggregates such as `gyro_rows_parity_mismatch_rows` and `gyro_rows_max_abs_row_error` when present.

## Build wiring snapshot

- `GGML_CPU_BACKEND_SUBDIR=ggml-gyroscopic` selects the gyroscopic CPU backend tree.
- `GGML_GYROSCOPIC=ON` wires GyroGraph/GyroLabe sources into the CPU backend; `GGML_GYROSCOPIC` on ggml-base adds `gyroscopic-base.c` and graph defines for llama token feed.
- `src/tools/gyroscopic/ops_build.py` can build `gyrolabe_native` and llama test binaries used by benchmarks.

## Runtime sanity points

- `--seed 123 --temp 0 --top-k 1 --top-p 1.0` keeps decode deterministic for parity checks.
- If parity fails, confirm `GyroMatMul stats` shows `qubec_calls>0` and inspect `GyroRows` parity fields.
- Use single-thread runs (`-t 1`) to isolate scheduling variation from arithmetic differences.

## 2026-04 incident notes

- Legacy `exact_rows_count` / `gyro_registry_tensor_has_exact_structure` guards are removed.
- Strict mode uses witness execution and parity counters rather than short-circuiting on a single SCR gate.
