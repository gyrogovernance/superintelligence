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
