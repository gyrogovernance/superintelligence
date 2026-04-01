# Gyroscopic aQPU - CPU

## What this folder is

Implementation of GyroLabe and GyroGraph: aQPU theory and kernel work applied to LLM inference on CPU. QuBEC is the computational object of aQPU. Integration uses a pinned `llama.cpp` snapshot; all CPU backend edits stay under `external/llama.cpp/ggml/src/ggml-gyroscopic/` (never vendor `ggml-cpu/`).

Relative docs:

- `docs/Gyroscopic_ASI_SDK_Quantum_Computing.md`
- `docs/theory/Gyroscopic_ASI_Specs_Formalism.md`
- `docs/theory/Gyroscopic_ASI_Holography.md`
- `docs/theory/QuBEC_Climate_Dynamics.md`
- `docs/references/Analysis_Gyroscopic_Multiplication.md`
- `docs/references/Analysis_CGM_Constants.md`

Operational detail (build flags, env vars, hook sites, file maps, reading order) lives in `log.md`. This file is scope and guides only. When analyzing those sources, use `agent_playbook.md` (what to extract, where to record it, implementation order, what to update).

## Scope lock

1. **Replace matrix multiplication with gyroscopic matrix multiplication** (`GGML_OP_MUL_MAT`, `GGML_OP_MUL_MAT_ID`, `GGML_OP_OUT_PROD`), routed through Gyrolabe kernels inside the existing ggml CPU execution path.
2. **Do not reimplement** what `llama.cpp` already solves well when it is correctness-neutral: threading, chunk traversal, tensor plumbing, allocators, graph construction, CLI, `llama-bench`, `perplexity`, and most repacking machinery. Reuse that chassis; swap the **math** at the kernel boundary.
3. **Study `ggml-gyroscopic/` file by file** to extract **optimization grammar** (tiling, register use, reduction style, load patterns, repack contracts). Do not treat stock kernel **semantics** as the destination.
4. **State claims precisely** using **exactness class** (see `log.md`) so GGUF interoperability and native QuBEC execution are not confused.

## Two kinds of fidelity (do not merge them)

- **A. Fidelity to gyroscopic math:** The operator executed is the gyroscopic operator (decomposition, invariants, structural intent).
- **B. Fidelity to a pretrained transformer:** The backend does not destroy practical model behavior on existing GGUF weights.

They are not automatically the same. GyroLabe's job is to maximize A on the chosen surfaces **while** keeping B measurable and honest (bench + perplexity / logits checks as appropriate).

The guiding question is not whether `MUL_MAT` will be replaced, because that remains the central scope. The question is how to maximize gyroscopic ownership of the in-scope math surfaces while keeping the bridge useful and the claims precise.

## Runtime ownership vs math ownership

- **Runtime** -- Owner: `llama.cpp`. Role: graph execution, scheduling, memory, model load, tokenizer, KV cache, most ops unchanged, harnesses.
- **Math (substitution)** -- Owner: Gyrolabe + thin hooks. Role: gyroscopic `MUL_MAT` / `MUL_MAT_ID` / `OUT_PROD`, support-based attention weighting (replacement path for softmax-like selection), structural extraction, future native packed formats.
- **Glue** -- Owner: `gyroscopic-bridge.*`. Role: env policy, strict/trace, call into `codec.c` (and linked gyrolabe TUs), fail-loud when required.

**Llama decides when and with what shapes; our code decides what numeric operator runs** at the hooked sites.

## Q8_0 and llama kernels: the precise line

Llama's Q8_0 inner loops are a strong **instruction-level** reference (integer carriers, AVX2 patterns, blocking). That does **not** mean leaving `MUL_MAT` stock and calling it gyroscopic.

**Do:** Reuse blocking, reduction grammar, repack-friendly traversal, and register discipline from `quants.c`, `arch/x86/quants.c`, `simd-gemm.h`, `repack.cpp`.

**Do not:** Confuse their quantized dot **semantics** with your finished gyroscopic operator, or skip replacement because the carrier looks "already integer."

Study and adapt means extracting loop structure, register strategy, reduction shape, repack contracts, and dispatch grammar. It does not mean inheriting stock numerical semantics where those semantics conflict with gyroscopic multiplication.

## Substitution priority (current)

**Own now (in scope):** gyroscopic matmul surfaces above, gyroscopic `OUT_PROD` where hooked, support-based weighting where wired, QuBEC-facing extraction in `scalar.c` / gyrolabe as used.

**Keep stock until explicitly promoted:** `RMS_NORM`, `ROPE`, `GELU`, `SILU`, and generic unary activations unless graph evidence and acceptance criteria say otherwise. Widening substitution before matmul is stable wastes validation budget.

**Architecture focus for study:** generic `ggml-gyroscopic` sources plus `arch/x86/*` only. Ignore `amx/`, `kleidiai/`, `spacemit/`, `llamafile/`, and non-CPU backends unless portability becomes a goal.

## Exactness (summary)

Claims must name the class. Full table and GGUFinterop notes: `log.md` -> **Exactness classes**.

In short: **kernel-exact** (aQPU byte law and discrete observables) is not the same as **deterministic-numeric** gyroscopic tensor ops over Q8_0 + fp16 scales + float accumulation. Both can be valid; they must be labeled.

## Non-negotiable rules

1. No OpenCL for this track. CPU only. AVX2 baseline for the intended x86 path.
2. No silent Python fallbacks when gyroscopic matmul (or other enabled gyro paths) is on.
3. Do not patch `ggml-cpu/`; integrate only under `ggml-gyroscopic/`.
4. Unsupported live paths: fail loudly when strict mode says so (see `log.md` for env).

## Verification (what to use, not how)

Use upstream tools with one binary and env toggles: `llama-bench`, `perplexity`, and `tests/export-graph-ops` to learn the real op surface for the target GGUF. Commands and bench env: `log.md`.

## Target model

Primary trace model: `Qwen3.5-4B-Q8_0.gguf` (path may be set in `config/gyroscopic_llm.yaml`). Used to validate coverage and behavior, not as a mandate to implement every other architecture in the tree.
