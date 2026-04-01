# Gyroscopic file analysis playbook

Use this when stepping through the **Examine first** list in `log.md` (paths under `external/llama.cpp/`). It does not replace `agent.md` (scope) or `log.md` (facts and wiring). It tells you **what to pull out**, **where it goes**, and **what to change** after reading.

## Before you open the file

1. Confirm you are in `ggml-gyroscopic/` (or the listed test/tool path), not vendor `ggml-cpu/`.
2. Note the **exactness class** you are implementing for this surface (`log.md` -> Exactness classes). Do not claim kernel-exact for Q8_0 + float accum paths unless the spec says so.
3. Decide the **output of this read**: inventory only, hook design, kernel rewrite plan, or bench procedure. One primary outcome per session avoids endless analysis.

## What to extract (categories)

For each file, tag findings into one or more of these buckets. Copy short quotes or symbol names into **`exports/<id>.md`** (raw extraction only) or **Findings** in `log.md`; do not paste huge slabs into git unless it is a committed design note. **Do not** put reading guides or step-by-step procedure in `exports/`; guides stay in `agent.md`, `agent_playbook.md`, and `log.md`.

- **Op / API contract** -- What: function names, tensor `ne`/`nb`, `src0`/`src1` roles, types, contiguity assumptions. Why: Gyrolabe and bridge must match ggml calling conventions or strict mode will fire.
- **Dispatch path** -- What: who calls whom: graph compute -> `ggml_compute_forward_*` -> vec_dot / repack / gemm. Why: you need to know every entry that can still bypass your kernel.
- **Orchestration** -- What: chunking, `ith`/`nth`, barriers, workspace (`wdata`) sizing. Why: usually keep stock; only change if gyro kernels need different chunk boundaries.
- **Microkernel grammar** -- What: load width, widen/madd pattern, horizontal sum, unroll depth, scale application point. Why: port into `codec.c` (or gyrolabe TU) as instruction shape, not as copied float semantics if gyro math differs.
- **Repack / extra buffer** -- What: `tensor_traits`, `forward_mul_mat`, when repack wins over generic path. Why: if repack bypasses `vec_dot`, hooks must cover repack or you silently run stock.
- **Bypass map** -- What: alternate fast paths, repack-only routes, dtype-specific kernels, fused branches that avoid the generic hook. Why: key to honest coverage claims for `MUL_MAT` and related ops.
- **Hook locus** -- What: exact function + line region for `#ifdef GGML_USE_GYROSCOPIC`. Why: prefer minimal call sites; extend `gyroscopic-bridge` instead of scattering logic.
- **Validation** -- What: which test or tool proves parity or acceptable drift. Why: tie each code change to `llama-bench`, `perplexity`, `export-graph-ops`, or `test-backend-ops` as appropriate.

## What to do with the knowledge

1. **If it is contract or bypass risk:** Add a one-line bullet under **Findings** in `log.md` (or a dated subsection you append there). Example: "repack `forward_mul_mat` used for Q8_0 when extra buft X".
2. **If it is kernel grammar:** Sketch the loop nest (M/N/K order, block sizes) in `exports/<id>.md` or scratch; then map it to a **single** target in `gyrolabe/codec.c` (or new gyrolabe C file) with the same structure and different inner operation.
3. **If it is a new hook site:** Implement thin redirect in `ggml-gyroscopic` -> `gyroscopic-bridge` -> `gyromatmul_*` / future `gyrolabe_*`; document env and strict behavior in `log.md` (Hook sites section).
4. **If it contradicts `agent.md` scope:** Fix `agent.md` in a small PR-sized edit, or flag the human; do not let drift accumulate in chat only.
5. **If it reveals a bypass:** record it explicitly in `log.md` under Findings or Coverage risk. Do not leave bypass knowledge only in chat or private notes.

## How to implement (order)

1. **Inventory before code:** List symbols and call edges relevant to `MUL_MAT` / `MUL_MAT_ID` / `OUT_PROD` (and any op you are promoting).
2. **Bridge first or kernel first:** Prefer **kernel contract in `core.h`** + stub, then bridge wire, then fill kernel. Avoid giant bridge blobs.
3. **One vertical slice:** One dtype/shape family working end-to-end with trace counters and a bench row beats many partial hooks.
4. **Strict mode:** Any new path either succeeds through gyro or aborts under strict; no silent stock fallback inside gyrolabe.

## What to update after implementation

- **`log.md`** -- Update when: new hook sites, env vars, CMake sources, findings from file reads, corrected "facts" rows.
- **`gyrolabe/core.h`** -- Update when: new exports, exactness class tag per symbol, block layout comments.
- **`gyroscopic-bridge.h` / `.cpp`** -- Update when: new entry points, counters, error strings.
- **`agent.md`** -- Update when: only if substitution priority or scope lock changes (rare).
- **Bench / config scripts** -- Update when: new modes, timeouts, or required trace markers.
- **`CHANGELOG.md` (repo root)** -- Update when: user-facing or milestone integration changes (project convention).

## Per-file extraction prompts

Use these as checklists. Check off only what applies.

### `tests/export-graph-ops.cpp`

- Extract: how graphs are reserved (`llama_graph_reserve`), pp vs tg token counts.
- Do: run or trace output for target GGUF; save op list to a small note or `log.md` **Findings**.
- Implement: none in this file; use output to order hook work.
- Update: `log.md` with "model X hits ops: ...".

### `ggml-gyroscopic/ops.cpp`

- Extract: `GGML_OP_*` branches for matmul, out_prod, softmax, norms; tensor layout comments in code.
- Do: mark which ops are in-scope for gyro substitution vs stock (`agent.md`).
- Implement: only thin hooks if policy says op-level entry; most matmul work stays in `ggml-cpu.c` / vec / repack today.
- Update: `log.md` if you add or move hook strategy at op layer.

### `ggml-gyroscopic/ggml-cpu.c`

- Extract: `ggml_compute_forward_mul_mat` and related chunk functions; `type_traits_cpu` and `vec_dot` routing; workspace layout.
- Do: map chunk boundaries to your GEMM API (row/col, ldc, batch dims).
- Implement: gyro path inside existing structure; preserve threading.
- Update: `log.md` Hook sites section and any new shape conditions for GEMM.

### `ggml-gyroscopic/ggml-cpu.cpp`

- Extract: `supports_op` rules for `MUL_MAT`, `OUT_PROD`, extra buffer types.
- Do: ensure gyro hooks do not violate support checks; adjust only if you add new buffer types.
- Update: `log.md` if behavior changes.

### `ggml-gyroscopic/arch/x86/quants.c` and `quants.c`

- Extract: inner loop of `ggml_vec_dot_q8_0_q8_0` (and any near variants); block struct layout; reduction tail.
- Do: list registers and steps; decide gyrolabe inner replacement preserving block boundaries.
- Implement: `codec.c` hot loops + existing bridge vec_dot hook.
- Update: `log.md` if layout asserts or hook conditions change.

### `ggml-gyroscopic/simd-gemm.h`

- Extract: tile sizes, microkernel interface, how `M`/`N`/`K` are peeled.
- Do: mirror tiling in `gyromatmul_gemm_*` or a new gyrolabe GEMM; drop naive triple loops when ready.
- Update: note in `log.md` under findings when GEMM strategy upgrades.

### `ggml-gyroscopic/repack.cpp` and `arch/x86/repack.cpp`

- Extract: when `forward_mul_mat` runs; `gemm`/`gemv` templates; dtype pairs.
- Do: list bypass risk: "generic vec_dot never called if ...".
- Implement: gyro route inside repack forward path or document strict abort until done.
- Update: `log.md` hook coverage and strict behavior.

### `ggml-gyroscopic/vec.cpp` and `vec.h`

- Extract: `ggml_vec_dot_f32` and helpers; hook already present for gyro.
- Do: identify other vec_* used on hot attention paths if you expand scope later.
- Implement: extend bridge only if new vec ops are gyro-owned.
- Update: `log.md` if new hooks.

### `tools/llama-bench/` and `tools/perplexity/`

- Extract: CLI flags relevant to CPU-only, threads, batch, model path; output format for scripting.
- Do: define a fixed command line for stock vs gyro env in bench script or `log.md`.
- Implement: changes only in repo scripts if needed to parse markers.
- Update: `log.md` bench section.

### `tests/test-backend-ops.cpp`

- Extract: how backends are compared; op test registration patterns.
- Do: decide if a gyrolabe-only unit test or this harness is the right parity check.
- Implement: rarely edit upstream; prefer local tests under `tests/gyroscopic/` if project adds them.
- Update: `log.md` if new verification command is standard.

## Session footer (copy into `exports/<id>.md` or PR)

- File(s) read:
- Extracted (1-3 bullets):
- Action: inventory / hook / kernel / bench
- Code touched:
- `log.md` updated: yes / no
- Exactness class for change:
