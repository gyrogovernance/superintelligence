# Gyroscopic run log

Dated gate / build notes. **Status: `runtime_NavPAD.md`.** This file is evidence only.

## 2026-08-07 — Arc 1 (MatMul) CLOSED

- Law: `Y = exact_Q1_0_q8 * manifold_gain`. Gate A PASS; Gate B mean cos 0.9999 + perturbation PASS; Gate C Paris 2520 disp lines PASS.
- Prompt suite 5/5; PPL stock 9.9156 ledger 9.9344 ratio 1.0019 PASS.

## 2026-08-08 — Arcs 2–3 (KV) CLOSED

- Arc 2B-2: Q8_0 K, `stock_score_calls=0`, PPL ratio 0.9992.
- Arc 3: Q8_0 V, `hqvm_attn_v_reduce`, PPL ratio 0.9989, V perturb collapses.

## 2026-08-09 — Forward pass incomplete (NOT a hybrid deliverable)

**Evidence only:** codecs PPL; causal perturb; residual/lift PPL; Norm live apply FAIL (`////////`).

**Correction:** “hybrid mode / hybrid milestone” was never consented. Incomplete forward stacks are unfinished §0 / eng debt — not a product mode. See `runtime_NavPAD.md` §0 / §6. OBSERVE/COMMIT/EMIT are scaffolding, not architecture.

## 2026-08-09 — Housekeeping

- SoT: `runtime_NavPAD.md` (no consented hybrid; §8 = stress flag stack only).
- `GYRO_SILU_CODEC=1` applies LUT (no `GYRO_SILU_COMMIT`).
- `run_bonsai` defaults to ledger+KV; `--incomplete-forward` for unfinished-site stress only.
- Helpers: `gates.py` / `run_bonsai` / `bench` / `build_*.ps1`; ledger → `ledger.py` (`ensure_ledger`).
- Gate names are architectural sites (`ledger` / `kv` / `codecs` / `causal` / `forward-probe`).
- Env: `incomplete_forward=` (no milestone nicknames as kwargs).
- C layout: `kernel` / `ledger` / `attn` / `codec`; traj+receipt types in `kernel`; one traj owner in `attn`; build via `build.py`.
- Verify: `ggml-cpu` clean rebuild + `llama-cli` link OK; `gates codecs --smoke-only` GATE_CODECS (smoke) PASS (Paris + `stock_score_calls=0`).

## 2026-08-09 — Arc 4 P0 hooks restore PASS

- Evidence: P0 restore + smoke PASS (parent). Forward-pass closure code follows: GyroClock L=36, signed Norm, residual law rename, embd/logits allowlist. Gate span/T*L results: deferred to parent rebuild+gate.

## 2026-08-10 — Native driver API (scaffold)

- `hqvm_forward_prefill` / `hqvm_forward_decode_step` / `hqvm_block_forward` in `layer.c`.
- Genealogy: `depth = t*36 + ell` at block boundary; not call-count.

## 2026-08-10 — Native Gate 0A / KV / ladders (executor integrity)

- **Implemented:** expanded inject summary counters; `request_begin` brackets; per-layer native KV write/read + null counters; `SET_ROWS` in stock-block bypass + early return under `hqvm_native_bypass_active()`; RoPE GQA fix (K head rope once, not ×gqa_ratio); `GYRO_ATTN_LEVEL` 0/1/2; `GYRO_FFN_LEVEL` 0/1/2; gates.py parses Gate 0A fields.
- **Evidence (`_build/native_L0_stderr.txt`, Attn-L0+FFN-L0):** prefill T=17 → `native_block_delta=612`, `K_writes=4896` (=17·36·8), all stock block counters 0, `set_rows_calls=0`, `kv_null_*=0`. Decode `native_block_delta=36` per step. `stock_tail_calls` rises (final norm + logits still stock).
- **Quality:** L0 stdout still gibberish (`weight truly gained reception…`) → Gate 3A FAIL; next debug is MatMul/norm/residual numerics, not dual-path/KV=NULL.
- **`gates causal` §4 native_driver:** PASS on expanded counters. Overall `GATE_CAUSAL` FAIL from holonomic `token_seq_differs` (sections 1–3), not native §4.


## 2026-08-09 — Arc 4 forward-pass closure gates

- `gates causal`: GATE_CAUSAL PASS. `depth_start=0 depth_end=864 steps=864 span=864` (`span==steps==T*L`, L=36). Lift perturb changes decode. Residual hits=720, `stock_score_calls=0`.
- Clock fixes in this pass: decode `token_pos` from seq cursor (not padded `Nk-1`); reset on longer prefill wave (`Nq >= seq_len` at layer 0); bump layer only when MVG lift ran.
- `gates codecs --smoke-only`: GATE_CODECS (smoke) PASS (Base/A/R/S/ARS Paris + `stock_score_calls=0`).
- `gates forward-probe`: GATE_FORWARD_PROBE PASS. Norm shadow `n_cos=40 mean_cos=1.0`. Micro-corpus PPL Base=12.0474 N_ratio=1.0123 H_ratio=1.0204 NH_ratio=1.0079 (ctx=128).
- `gates ledger`: GATE_LEDGER PASS. Prompts 5/5; stock_ppl=9.9156 ledger_ppl=9.9344 ratio=1.0019.
- `gates kv` (paris-only): GATE_KV PASS (GATE_KV_K + GATE_KV_V; Gate I long-context timeout raised to 900s).
