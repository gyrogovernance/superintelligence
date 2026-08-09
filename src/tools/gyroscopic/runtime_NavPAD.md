# Gyroscopic Runtime NavPad

**Single source of truth** for §0 status and unfinished work. Not a run log.

Evidence only: `log_NavPad.md`. Theory reading: `Analysis_Arcs_1_4_Theory.md`.

---

## 0. Goal (unchanged) and honest status

**§0 completion (intent):** the forward pass *is* a deterministic hQVM trajectory on Ω (|Ω| = 4096): Input → Controller → Tape → gyration → Shell, datatype **[Anchor, Restriction, Depth, Phase]**. Continuous magnitude only where aperture Δ > 0 forces a residual channel. Carrier is causal. Inference is gyroscopic-native — not a stock transformer with optional flags.

| Site | Scope | Status vs §0 |
|---|---|---|
| **Arc 1** | Temporal weight MatMul | **CLOSED** (real displace) |
| **Arcs 2–3** | Attention KV memory | **CLOSED** (real displace) |
| **Forward pass** | Softmax / RoPE / Norm / FFN / residual / lift / receipts | **NOT CLOSED** |

**There is no consented “hybrid mode.”** Nobody approved a permanent stock+gyro mix as a product, architecture, or §0 deliverable. Incomplete forward-site work (lift writing traj, residual Δ-modulation, codec LUTs beside stock ops) is **unfinished §0 / engineering debt**. Treating that stack as a milestone to ship, a named mode, or “good enough” is **scope creep and revisionism**. The only completion criterion remains §0 native inference.

**What exists today (incomplete forward stack, not a deliverable):** allowlisted Q1_0 MatMuls + Q8 KV + holonomic QK/Attn@V + flagged aperture/RoPE/SwiGLU + CGM-lift traj + residual add modulated by traj (`1+Δ·m`). That proves some coupling; it does **not** close §0.

Target: Bonsai-8B-Q1_0. Hooks: `external/llama.cpp/ggml/src/ggml-gyroscopic/`. Logic: `src/tools/gyroscopic/`.

Env: `production_gyroscopic_env(holonomic_kv=True)` = closed MatMul+KV (ensures ledger). Chat: `run_bonsai`. Ledger: `ledger.py` (`ensure_ledger` / `write_ledger`). Flag stacks that combine unfinished forward sites are stress/debug only (§8) — not product modes.

Gate / CLI / env names are **architectural sites** only (`ledger` / `kv` / `codecs` / `causal` / `forward-probe`, `incomplete_forward=`). Do not invent milestone nicknames as identifiers.

---

## 0.1 Process note — scaffolding labels are not architecture

Labels such as **OBSERVE / EMIT / DISPLACE / COMMIT**, and flag pairs like `GYRO_NORM_CODEC` vs `GYRO_NORM_COMMIT`, were **progressive engineering attempts** to shadow-then-flip ops. They are **not** part of the CGM/hQVM architecture and must not be preserved as doctrine.

**“Hybrid” is the same class of mistake:** an eng nickname for “stock graph still runs; some sites hooked.” It is not a datatype, not a carrier law, and not a consented deliverable. Do not grow product surface around it.

Owned sites apply when their flag is on (`GYRO_SILU_CODEC=1` applies the SwiGLU LUT; no separate COMMIT). What the theory requires: each site either (a) is compiled into the datatype / trajectory law, or (b) retains an explicit Δ-forced continuous residual with a stated residual law — not a permanent observe/hybrid forever mode.

---

## 1. Contract

1. Kernel math: `kernel.c` / `kernel.h`. Reverse-compiler d=6: `ledger.c` (`step_uv6`). Do not conflate.
2. No Python in the hot path. Python = export and gates only.
3. No assist-BLAS / bias decoration. The kernel owns the substituted site.
4. Verify with **canonical gates** (§7), not diagnostic branches on production paths.
5. Production ledger (`HQVMLEDS`) does not duplicate GGUF weights. Signs/scales from ggml RAM.
6. `GYRO_LEDGER_STRICT=1`: allowlisted site that cannot displace aborts.
7. Do not declare §0 closed on the basis of incomplete mixed-path stacks.

---

## 2. Datatype and split

**[Anchor, Restriction, Depth, Phase]** — Anchor `(u6,v6)`, `chi6 = u6 ⊕ v6`; Restriction = shared byte table; Depth = ledger time; Phase = selected byte (plus family bits when forming a full intron).

**Manifold vs controller:** manifold commits *which* shell; controller commits *how much*.

**Δ ≈ 0.0207:** residual individuality is not disposable. Where continuous content is load-bearing, it survives as an explicit residual channel — not as “we left stock ops forever.” (A Δ-forced residual channel is theory; a permanent transformer+hooks “hybrid mode” is not.)

**Byte = q6 + family:** `intron = byte ⊕ GENE_MIC_S`. Family from `phase_idx mod 4` when emitting.

---

## 3. Weight MatMul (Arc 1) — CLOSED

**Law.** `Y = exact_Q1_0_q8_dot_product × manifold_gain`. Families today: `attn_q/k/v/output`, `ffn_gate/up/down` (2520 sites).

**Env.** `GYRO_LEDGER_PATH`, `GYRO_LEDGER_STRICT`, optional `GYRO_LEDGER_ALLOW`.

**Boundary (not Arc-1 law failure):** `token_embd.weight` is embedding `GET_ROWS` (not a Dense MatMul); `output.weight` is logits `MUL_MAT`. Both are Q1_0 and **outside** the allowlist — stock path. Deferred engineering cut, **not** a user-declared end design. Native inference requires a decision: compile under Arc 1 law where applicable, or state an explicit embd/logits residual boundary with theory.

---

## 4. Attention KV (Arcs 2–3) — CLOSED

**Key.** `GYRO_KV_KQ8=1` → Q8_0 K; F16 K never allocated; holonomic in-place scores.

**Value.** `GYRO_KV_V=1` → Q8_0 V; `hqvm_attn_v_reduce`.

**Env.** `GYRO_HOLONOMIC_ATTN=1`, `GYRO_KV_KQ8=1`, `GYRO_KV_V=1`.

**Closed on:** Q8_0 as the owned magnitude chart for K/V. Softmax/RoPE/Norm/FFN/residual were never Arc 2–3 scope (forward pass, §5). Do not reopen Arc 3 with a speculative “Value-beyond-Q8” unfinished item unless theory explicitly revisits V.

---

## 5. Temporal forward pass — OPEN vs §0

### 5.1 What exists (incomplete inventory — not accepted deliverables)

| Site | Today | vs §0 |
|---|---|---|
| Softmax | Aperture mix live when flagged; PPL PASS | Partial — still stock exp + ε; not traj-native attention bookkeeping |
| RoPE | 256-tick LUT live when flagged; PPL PASS | Partial — not bound to causal phase / receipt depth |
| SwiGLU | LUT live when flagged; PPL PASS | Partial — chart of SiLU, not FFN-as-trajectory |
| RMSNorm | Stock scale; Δ-ruler encode exists but **broken** for live apply (unsigned clamp → `////////`) | Open — must compile gain under signed Δ-ruler + residual law |
| Residual add | Stock F32 + **Δ-modulation** from lift traj | Partial — causal touch, **not** ledger-depth residual stream |
| CGM-lift | One byte/layer; traj advances; residual reads it | Partial — write/read exist; model still not traj-native |
| Receipts | Can seal from lift traj | Partial — not one Moment genealogy across prefill/decode as native time |
| χ | Write-time `k_chi6` for lift; aperture can use store | Partial — unify everywhere; no Q8 re-derive as primary |

### 5.2 Unfinished work (theoretically required for native inference)

These reopen **§0 / forward-pass sites only**. Arcs 1–3 stay closed unless a site forces revisiting them.

**A. Carrier and time**

1. Make the trajectory the **primary** control of depth/phase for every compiled site — not a Δ-sidecar on stock adds.
2. Define residual-stream law: continuous Δ-channel beside carrier **or** compile accumulation into ledger depth; today’s `1+Δ·m` on F32 add is unfinished bridge code, not an accepted residual architecture.
3. One Moment / receipt genealogy across prefill and decode (replayable native time).
4. Lift schedule: one byte/layer is a choice; native may need richer emission (heads/tokens) with thread-safe traj.

**B. Continuous chart → coordinates + residual**

5. **RMSNorm:** fix signed Δ-ruler encode/decode (negative `log2(g/g0)` must not clamp to q=0); apply live; measure **rel magnitude error**, not cosine; keep only Δ-forced residual of the gain — no permanent shadow mode as architecture.
6. **Residual / skip connections:** finish beyond modulation — either a stated Δ-forced continuous residual law or compile into carrier depth.
7. **RoPE:** bind finite turn chart to trajectory phase / depth (not only LUT of stock angles).
8. **Softmax:** keep exp as magnitude decoder; aperture as distribution constraint; bind rank/χ to write-time ledger χ under causal carrier (not re-derived ad hoc).
9. **FFN / SwiGLU:** causal compile or residual+coordinate on carrier — LUT alone is not native FFN.
10. Drop remaining scaffolding dual-paths once a site is owned (Norm still has codec/commit debt). Retire “hybrid” naming from eng surfaces when forward sites are actually owned.

**C. Boundaries and identity**

11. **`token_embd` / `output.weight`:** decide embd (`GET_ROWS`) and logits (`MUL_MAT`) under Arc 1 law or explicit theory-backed residual — not silent stock.

**D. Rejected (do not revive as unfinished)**

- χ-only / Khat as live QK energy
- Percolation / shell-λ as softmax weights
- Aperture ε-mixture on RoPE / Norm / SiLU
- Dropping Norm residual wholesale
- Invented audit fronts (generic Dense/bias inventory; Value-beyond-Q8 as §0 debt while Arc 3 stays closed)
- **Shipping or documenting a permanent “hybrid mode” as §0 success**

### 5.3 Incomplete causal bridge (stress evidence only — not final law)

- Lift writes traj once per layer (holonomic FA guard).
- Residual path reads `state24` → `gain = 1 + Δ·(shell−3)/3` on same-shape F32 4096 adds.
- Perturb (`GYRO_CGM_LIFT_PERTURB`) changes decode — proves coupling, **not** native replacement and **not** consent for a hybrid product.

---

## 6. Rejected paths

| Path | Why |
|---|---|
| χ-only / Khat as live QK | Discards magnitude |
| Percolation / shell-λ as softmax weights | Wrong category |
| Aperture ε on RoPE / Norm / SiLU | Wrong site |
| Drop Norm residual wholesale | Collapses depth |
| Dense/bias/add “inventory and classify” as open front | Not a Bonsai site gap; biases absent; ADD is §5.2 A.2/B.6; embd/logits are §5.2 C.11 |
| Value-beyond-Q8 as §0 unfinished | Arc 3 closed on Q8_0; speculative reopen only |
| Treating incomplete forward stacks as §0 done | Revisionism |
| Consenting to a permanent “hybrid mode” as architecture or deliverable | Scope creep; not requested |
| Preserving OBSERVE/EMIT/COMMIT as architecture | Scaffolding, not theory |

---

## 7. Canonical gates (one module)

```
python -m src.tools.gyroscopic.helpers.gates ledger
python -m src.tools.gyroscopic.helpers.gates kv [--ppl]
python -m src.tools.gyroscopic.helpers.gates codecs [--smoke-only]
python -m src.tools.gyroscopic.helpers.gates causal
python -m src.tools.gyroscopic.helpers.gates forward-probe
```

| Subcommand | Role |
|---|---|
| `ledger` | MatMul displace (Paris + PPL) |
| `kv` | Q8 K then V + holonomic Attn@V |
| `codecs` | Aperture/RoPE/SwiGLU vs Base (site probes) |
| `causal` | Lift perturb changes decode — coupling proof only |
| `forward-probe` | Norm shadow + residual/lift PPL (Norm COMMIT may FAIL until §5.2.5) |

Prefer smoke → causal → one cached Base PPL → variants. Do not re-Base every flag flip.

---

## 8. Incomplete forward flag stack (stress/debug only — not a product mode)

Code: `production_gyroscopic_env(incomplete_forward=True)` / `run_bonsai --incomplete-forward`. Use only to reproduce unfinished-site measurements. Not architecture.

```
GYRO_LEDGER_PATH=.../hqvm_sidecar.bin
GYRO_LEDGER_STRICT=1
GYRO_HOLONOMIC_ATTN=1
GYRO_KV_KQ8=1
GYRO_KV_V=1
GYRO_APERTURE_SOFTMAX=1
GYRO_ROPE_CODEC=1
GYRO_SILU_CODEC=1
GYRO_CGM_LIFT=1
GYRO_RESIDUAL_HYBRID=1
```

Do not set `GYRO_NORM_COMMIT`. Next theory-facing work: §5.2 (Norm signed ruler first among continuous ops; then residual law; then embd/logits; then retire scaffolding and “hybrid” aliases).

Rebuild: `cmake --build external/llama.cpp/build --config Release --target ggml-cpu --clean-first` then `llama-cli` / `llama-perplexity`.

---

## 9. File map

| Role | Path |
|---|---|
| Status (this doc) | `runtime_NavPAD.md` |
| Evidence | `log_NavPad.md` |
| Theory notes | `Analysis_Arcs_1_4_Theory.md` |
| C linked into ggml-cpu | `kernel.c`, `ledger.c`, `attn.c`, `codec.c` (+ headers) |
| Public include umbrella | `api.h` → `constants` / `kernel` / `ledger` / `attn` / `codec` |
| Thin hooks | `external/llama.cpp/ggml/src/ggml-gyroscopic/` |
| Env | `config.production_gyroscopic_env` (auto-`ensure_ledger`) |
| Build | `build.py`; `helpers/build_*.ps1` |
| Ledger (Python) | `ledger.py` — thin HQVMLEDS; companion to `ledger.c` |
| Gates / chat / bench | `helpers/gates.py`, `helpers/run_bonsai.py`, `helpers/bench_gyroscopic_llama.py` |

One trajectory instance lives in `attn.c` (lift); traj/receipt types and steppers live in `kernel`. Continuous charts live in `codec`. Do not reintroduce split underscored modules.
