# Gyroscopic Runtime NavPad

**Plan, verified status, and ownership boundary** for gyroscopic ASI inference on Bonsai-8B-Q1_0.

Dated evidence lives in `log_NavPad.md`. The laws and their theoretical justification live in `Analysis_Gyroscopic_Runtime.md`. The wider theory set begins with `docs/Gyroscopic_ASI_Foundations.md`, `docs/specs/hQVM_Specs_Formalism.md`, `docs/specs/hQVM_QuBEC_Theory.md`, and `docs/specs/Gyroscopic_ASI_Runtime_Specs.md`.

---

## 1. Goal

Implement full gyroscopic ASI inference in which the forward pass is a deterministic hQVM trajectory on Ω, with |Ω| = 4096:

Input → Controller → Tape → gyration → Shell

The runtime datatype is **[Anchor, Restriction, Depth, Phase]**. One request has one **Genealogy**, consisting of the law, the request-scoped Common Source anchor, and depth.

The intended native algebra is integer exact or dyadic exact. Euclidean floating-point arrays may remain as temporary interoperability charts during development, but they are not the final architecture. Transcendental decision laws must resolve to finite charts, popcount, polynomial-in-λ weighting, or algebraic selection. The aperture invariant is represented by the Formalism tick `Q_256(Δ) = 5/256`.

**Full completion criterion:** every load-bearing site, including embedding lookup, all 36 transformer blocks, final normalization, and output projection, is governed by the datatype and Genealogy without stock semantic ownership. A separate completion criterion is a genuinely float-free hot path.

Target model: Bonsai-8B-Q1_0. Thin hooks: `external/llama.cpp/ggml/src/ggml-gyroscopic/`. Native logic: `src/tools/gyroscopic/`.

---

## 2. Runtime contract

1. Kernel math belongs in `kernel.c` and `kernel.h`. Reverse compilation at d=6 belongs in `ledger.c` through `step_uv6`. These are distinct layers.
2. Python is limited to export, orchestration, and gates. It is not part of the inference hot path.
3. `HQVMLEDS` compiles weight law and does not duplicate GGUF weight payloads. Signs and scales are read from ggml RAM.
4. `GYRO_LEDGER_STRICT=1` makes failure to displace an allowlisted site fatal.
5. Genealogy is request-scoped and continuous across prefill and decode. Receipts are optional coordinates on that history.
6. Prefill is layer-major. Layer ℓ must produce K and V for every prompt token before layer ℓ+1 consumes the resulting residual stream.
7. Graph positions are authoritative for sequence lifecycle. Native decode must not infer absolute position from call count.
8. Native Q8_0 activation quantization must match ggml exactly because scale rounding and integer tie behavior affect parity and therefore manifold routing.
9. Passing tests is evidence of a law, not a substitute for the law.

---

## 3. Datatype and clocks

**Anchor:** `(u6,v6)`, with `chi6 = u6 ⊕ v6`.

**Restriction:** the shared byte table.

**Depth:** `depth = t·L + ℓ`, with `L = 36` for Bonsai.

**Phase:** the selected byte plus family bits when a full intron is formed.

**Family:** `fam = depth & 3`.

**Byte composition:** `intron = byte ⊕ GENE_MIC_S`.

**Manifold and controller:** the manifold commits which shell is selected. The controller commits how much amplitude is transported through exact Q1_0 by q8 products and dyadic laws.

| Object | Role |
|---|---|
| `HQVMLEDS` sidecar | Compiled weight law for MatMul |
| Genealogy | Request anchor, depth, phase, and transport history |
| Native KV plus `k_chi6` | Causal memory and its matching shell coordinate |
| Receipts or QR | Optional observation, outside the hot path |

---

## 4. Verified ownership state, 2026-08-10

### 4.1 What is now owned

The full Bonsai transformer block stack is natively driven under the default production laws.

| Site | Native law or behavior | Verified status |
|---|---|---|
| Block scheduling | `hqvm_forward_prefill`, `hqvm_forward_decode_step`, `hqvm_block_forward` | OWNED |
| Weight MatMul | exact Q1_0 by q8 controller product times manifold gain | OWNED in all 36 blocks |
| Q, K, V and output projections | native ledger MatMul | OWNED |
| Q and K per-head RMSNorm | Δ-ruler with Bonsai epsilon `1e-6` | OWNED |
| RoPE | Qwen3 normal consecutive-pair layout on the turn-tick chart | OWNED |
| K and V memory | native per-layer Q8_0 cache plus `k_chi6` | OWNED |
| Attention weighting | full λ^N law at `GYRO_ATTN_LEVEL=2` | OWNED |
| FFN and SwiGLU | family by occupation gate at `GYRO_FFN_LEVEL=2` | OWNED |
| Residual stream | `x ← x + y·(1 + Δ·m)` | OWNED |
| Genealogy | `depth=t·36+ℓ`, one continuous prefill/decode history | OWNED |
| CS lift | request anchor derived from embeddings and applied at injection | OWNED at current boundary |
| Stock transformer-block execution | bypassed, including final-layer row selection | ZERO observed calls |

The canonical native run produced valid text, including `The capital of France is Paris.`. The causal gate passed with a nonempty answer, a changed decode under lift perturbation, and a genealogy span equal to `T·L`.

Observed production-path counters included:

- `stock_block_forward_calls=0`
- `stock_flash_attn_calls=0`
- `stock_softmax_calls=0`
- `stock_rope_calls=0`
- `stock_rmsnorm_calls=0` inside bypassed blocks
- `stock_swiglu_calls=0`
- `stock_silu_calls=0`
- `stock_add_calls=0`
- `set_rows_calls=0`
- `kv_null_reads=0`, `kv_null_writes=0`
- `native_block_delta=36` for each decode token
- `K_writes=V_writes=chiK_writes=T·36·8` during prefill
- `pi_applied=1`

### 4.2 What is not yet owned

We can confidently claim **native ownership of the 36-block forward trajectory**. We cannot yet claim ownership of the entire model graph, and we cannot claim a float-free hot path.

| Boundary | Current state | Required closure |
|---|---|---|
| Embedding lookup | ggml supplies the initial F32 embedding row; native CS and Pi begin there | Own lookup and native chart entry |
| Final RMSNorm | stock tail executes after native residual injection | Move final norm into native driver |
| `output.weight` projection | stock tail produces logits | Move logits projection and ledger law into native driver |
| Sampling | llama.cpp sampler remains the consumer of logits | Declare whether sampling is chassis or architecture scope |
| Float-free execution | native blocks still use F32 residual, Q/K/V scratch, attention accumulation, decoded scales, and some chart construction | Replace temporary Euclidean charts with integer or dyadic storage and arithmetic |
| Perplexity acceptance | valid generation and causal gates pass; production native PPL has not been accepted | Run cached-base PPL only after the tail boundary is settled |

`stock_tail_calls=7` in the eight-token smoke run is therefore expected evidence of the remaining final norm and logits boundary, not evidence that stock transformer blocks executed.

**Ownership verdict:** block-forward ownership is closed. Whole-graph ownership, float-free ownership, and PPL closure remain open. The project has reached valid native text generation, but it has not yet reached every completion criterion stated in Section 1.

---

## 5. Engineering knowledge that closed native generation

### 5.1 Qwen3 graph topology is part of the executable contract

Bonsai is a Qwen3-family model. The native driver must respect these facts:

- Q and K receive per-head RMSNorm before RoPE.
- RoPE uses the normal consecutive-pair convention `(2i, 2i+1)`, not the NeoX half-split convention.
- The graph contains `inp_out_ids` row selection inside the final transformer block.
- Final tensors are named `result_norm` and `result_output`.
- Native residual injection must target the input of final RMSNorm, not its output.
- The unnamed I32 input consumed by RoPE carries authoritative absolute positions.

The final-layer `GET_ROWS` operation was load-bearing. Allowing it to execute after native injection replaced the native residual with stale graph data. It is now classified as stock block work whenever its source belongs to a transformer block.

### 5.2 Prefill causality is layer-major

A token-major prefill computes later layers before all same-layer prompt K and V entries exist. Correct causal memory requires:

```text
for layer ℓ:
    for prompt token t:
        execute block(t, ℓ)
```

Only after all prompt tokens complete layer ℓ may the residual stream advance to layer ℓ+1.

### 5.3 Sequence lifecycle follows model positions

The graph position tensor determines request reset, prefill, and decode position. Native KV is reset when a new sequence starts or graph positions rewind. This prevents prior requests and guessed positions from contaminating Genealogy and causal memory.

### 5.4 Q8_0 compatibility is semantic

The native q8 quantizer now matches `quantize_row_q8_0_ref` by using `roundf` and by storing the block scale after FP16 rounding. This is not only numerical compatibility. The quantized activation enters mismatch parity and chirality decisions, so a one-bit discrepancy can choose a different shell.

### 5.5 Norm has two distinct reference moments

The inverse RMS gain is dimensionless and is encoded around reference 1. The learned normalization weights are encoded around their tensor-local geometric mean. Using one reference for both conflates two moments and distorts the ruler.

The fixed RMS moment also has a finite-word-size law. Direct Q16 squaring at hidden width 4096 overflows signed 64-bit accumulation once late-layer residual magnitudes become validly large. The corrected method normalizes by `amax`, accumulates bounded Q15 squares, and restores scale afterward. This preserves the fixed moment without overflow.

### 5.6 The decisive closure chain

Valid native generation required all of the following to hold together:

1. Qwen3 consecutive-pair RoPE.
2. Bit-compatible Q8_0 activation quantization.
3. Layer-major prefill.
4. Position-driven sequence lifecycle.
5. Correct `result_norm` input injection.
6. Final-block `GET_ROWS` bypass.
7. Separate RMS and learned-weight references.
8. Overflow-safe fixed RMS accumulation.

No manifold law was disabled to obtain the passing result.

---

## 6. Remaining work

### 6.1 Close whole-graph ownership

Move final RMSNorm and `output.weight` projection into the native driver. Decide explicitly whether embedding lookup and sampling are part of architectural ownership or declared chassis boundaries. Update counters so a completed whole-graph run has no unexplained stock tail work.

### 6.2 Close the native algebra

Inventory every F32 field and operation in `layer.c`, `attn.c`, `codec.c`, and `ledger.c`. Replace them site by site with declared integer or dyadic charts. Priority order:

1. Residual and normalized activation storage.
2. Q, K, V and FFN scratch buffers.
3. Attention score and value accumulation.
4. Q1_0 scale decode and controller gain representation.
5. RoPE tick application without F32 row buffers.
6. Native final norm and logits.

A zero stock-op counter does not prove float-free execution. Float-free acceptance requires datatype and operation audits plus dedicated counters or compile-time enforcement.

### 6.3 Quality closure

After whole-graph ownership is fixed, run the cheapest production-native perplexity sample against a cached stock baseline. Broaden the corpus only if the small gate is sound. Keep stress tests last.

### 6.4 Diagnostic hygiene

Keep law-level counters and request summaries. Remove temporary numerical probes from default output once they no longer provide acceptance evidence.

---

## 7. Canonical gates

```powershell
python -m src.tools.gyroscopic.helpers.gates ledger
python -m src.tools.gyroscopic.helpers.gates kv --ppl
python -m src.tools.gyroscopic.helpers.gates codecs --smoke-only
python -m src.tools.gyroscopic.helpers.gates causal
python -m src.tools.gyroscopic.helpers.gates forward-probe
```

| Subcommand | Role |
|---|---|
| `ledger` | MatMul displacement and quality |
| `kv` | Q8 K/V and holonomic attention memory |
| `codecs` | Aperture, RoPE, and FFN site probes |
| `causal` | Native driver, Genealogy span, and lift perturbation |
| `forward-probe` | Norm and residual measurements |

Preferred order: smoke, causal, one cached Base PPL, then variants.

---

## 8. Default production law stack

`production_gyroscopic_env(holonomic_kv=True)` enables the native driver and owned site laws. Native internal defaults are:

```text
GYRO_ATTN_LEVEL=2
GYRO_FFN_LEVEL=2
GYRO_NATIVE_KV=q8
GYRO_NATIVE_ROPE=tick
GYRO_NATIVE_NORM=Delta-ruler
GYRO_NATIVE_RESIDUAL=Delta-law
```

Debug overrides such as `GYRO_NATIVE_KV=f32`, `GYRO_NATIVE_ROPE=float`, `GYRO_NATIVE_NORM=plain`, identity residual, stock softmax, or reduced attention and FFN ladders are diagnostic controls. They are not production architecture.

`production_gyroscopic_env(incomplete_forward=True)` is measurement-only and must not be described as the product path.

---

## 9. File map

| Role | Path |
|---|---|
| Status and plan | `runtime_NavPAD.md` |
| Dated evidence | `log_NavPad.md` |
| Theoretical exposition | `Analysis_Gyroscopic_Runtime.md` |
| Native block driver | `layer.c`, `layer.h` |
| Kernel and transport | `kernel.c`, `kernel.h` |
| Ledger MatMul | `ledger.c`, `ledger.h`, `ledger.py` |
| Attention and finite charts | `attn.c`, `attn.h`, `codec.c`, `codec.h` |
| Thin graph hooks | `external/llama.cpp/ggml/src/ggml-gyroscopic/` |
| Production environment | `config.production_gyroscopic_env` |
| Gates and runner | `helpers/gates.py`, `helpers/run_bonsai.py` |

One trajectory instance lives in `attn.c`. Trajectory and receipt types live in the kernel. Finite and dyadic charts live in the codec.
