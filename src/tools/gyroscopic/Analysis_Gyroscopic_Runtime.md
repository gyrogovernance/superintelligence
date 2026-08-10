# Analysis: Gyroscopic Native Inference

This document explains why the gyroscopic runtime on Bonsai-8B-Q1_0 has its present mathematical form. It is a theoretical exposition, not an engineering log. Runtime ownership and acceptance evidence belong in `runtime_NavPAD.md` and `log_NavPad.md`.

Primary theory sources are `docs/Gyroscopic_ASI_Foundations.md`, `docs/specs/hQVM_Specs_Formalism.md`, `docs/specs/hQVM_QuBEC_Theory.md`, `docs/specs/Gyroscopic_ASI_Runtime_Specs.md`, and the associated CGM, Mass Gap, Percolation, Climate, and Gyroscopic Multiplication analyses.

---

## 1. Thesis

A frozen pretrained transformer can be compiled into gyroscopic-native inference when its forward map is treated as a deterministic hQVM trajectory on the finite manifold Ω, with |Ω| = 4096:

Input → Controller → Tape → gyration → Shell

The unit of runtime state is:

**[Anchor, Restriction, Depth, Phase]**

A model such as Bonsai supplies a learned operator field. hQVM supplies the native state space, transport law, causal clock, and exactness classes through which that field is evaluated. The aim is not to place a second symbolic process beside the transformer. The aim is to compile each load-bearing transformer site into one coherent trajectory governed by the same datatype and Genealogy.

Magnitude belongs to the QuBEC exactness classes of integer exact or dyadic exact. Euclidean floating-point arrays can serve as temporary coordinate charts, but they are not the native semantics. Transcendental decision laws such as `exp`, `sqrt`, sigmoid, and continuous trigonometric phase are signs that the computation has been expressed in an external chart. On Ω, distance becomes popcount, occupation weighting becomes polynomial in λ, phase becomes a finite turn chart, and selection becomes algebraic.

Aperture Δ is the irreducible phase defect of depth-four Balance Universal. Its formal dyadic coordinate is:

`Q_256(Δ) = 5/256`

Genealogy is the time axis of inference. It consists of law, request-scoped Common Source anchor, and depth. Prefill and decode are not separate histories. They are two execution regimes of one causal trajectory.

---

## 2. The native datatype

### 2.1 Anchor

The carrier anchor is `(u6,v6)` with chirality:

`χ = u6 ⊕ v6`

The anchor identifies the local state on Ω. Chirality is the transport register on `GF(2)^6`. Its shell occupation is:

`N = popcount(χ)`

The shell is the native radial coordinate. It replaces the assumption that Euclidean vector norm is the only meaningful notion of distance or magnitude class.

### 2.2 Restriction

Restriction is the shared byte table that constrains admissible transport. It is not a learned dense matrix. It is the finite law through which a requested shell and a current anchor determine a legal byte transition.

### 2.3 Depth

For Bonsai, which has 36 transformer layers, depth is:

`depth = t · 36 + ℓ`

Here `t` is absolute token position and `ℓ` is layer index. This depth is the causal clock. A forward pass must therefore advance once for every `(token, layer)` pair and must preserve continuity from prompt prefill into autoregressive decode.

### 2.4 Phase and family

The byte is composed from a six-bit transport payload and a two-bit K4 family deck:

`intron = byte ⊕ GENE_MIC_S`

`fam = depth & 3`

Family is not decorative metadata. It is the spinorial phase layer of the depth-four cycle. It turns a transport payload into a legal phase-bearing byte and places the carrier on the CS, UNA, ONA, BU closure sequence.

### 2.5 Genealogy and ledger

Genealogy and `HQVMLEDS` are different mathematical objects.

- Genealogy is runtime history: law, Common Source anchor, depth, and phase.
- `HQVMLEDS` is compiled weight law: the finite sidecar needed to evaluate learned Q1_0 operators through the manifold.

Identifying the ledger with Genealogy would erase the distinction between a frozen operator and the history of a request acted upon by that operator.

---

## 3. Manifold and controller

Every compiled contraction has two complementary aspects.

**Manifold:** decides which shell and transport class are selected through XOR geometry, chirality, gyration, and restriction.

**Controller:** decides how much amplitude is transported through bipolar signs, integer accumulation, Q1_0 scales, q8 activations, and dyadic gain.

The joint MatMul law is:

`Y = exact_Q1_0_q8_dot_product × manifold_gain`

with:

`manifold_gain = 1 + Δ · mean(sign(parity(mismatch) XOR χ'_0))`

The exact controller product preserves the learned magnitude structure. The manifold gain realizes the selected gyration on the numeric output. Neither aspect can replace the other. Chirality without controller amplitude collapses learned magnitude. Controller amplitude without manifold routing leaves the native datatype causally inert.

This explains why Q8_0 compatibility is semantic rather than merely approximate. The quantized activation participates in mismatch parity. A different rounding tie or a scale stored in a different precision can alter a parity bit, select a different shell, and change the trajectory. Bit-compatible quantization is therefore part of the compiled law.

---

## 4. Aperture, occupation, and polarization

Depth-four Balance Universal cannot close losslessly in both temporal directions while retaining distinguishable outcomes. The residual phase defect is:

`Δ = 1 - ρ ≈ 0.0207`

The positive aperture permits common origin and individual excitation to coexist. Formalism places it on the fractional dyadic chart `T_256^(frac)` as `5/256`.

The one-cell partition law is polynomial:

`Z_1(λ) = 64 · (1 + λ)^6`

with shell occupation:

`π_λ(N) = C(6,N) · λ^N / (1 + λ)^6`

The native control variable is λ. Writing `λ = exp(-β)` is an optional lift into an external thermodynamic chart. It is not necessary to the native occupation law.

Polarization is:

`m = (E[N] - 3)/3 = (λ - 1)/(λ + 1)`

For one carrier state it becomes:

`m = (N - 3)/3`

This is the signed displacement between equality and complement horizons. It is the natural scalar through which shell individuality modulates a residual update.

---

## 5. What Bonsai and Qwen3 contribute to the theory

A native compiler cannot treat transformer architecture as an unordered bag of matrices. The pretrained topology is the boundary condition of the learned operator field. Gyroscopic compilation changes the chart and the site laws, but it must preserve the causal incidence relations learned by the model.

Bonsai is a Qwen3-family model with 36 layers, hidden width 4096, FFN width 12288, head dimension 128, 32 query heads, and 8 key/value heads. Its grouped-query ratio is therefore 4. Its normalization epsilon is `1e-6`. Its rotary parameters include frequency base `10^6`, YaRN frequency scale `1/4`, original context 16384, and the Qwen3 correction interval determined by beta-fast 32 and beta-slow 1.

These values are not arbitrary implementation constants. They define the pretrained coordinate field that must be mapped into native charts.

### 5.1 Per-head Q and K normalization

Qwen3 normalizes each query and key head before rotary transport. The theoretical order is therefore:

`residual → learned projection → head-local normalization → phase transport → causal pairing`

Head-local normalization establishes the scale class in which phase transport acts. Moving or omitting it changes the operator being compiled. In gyroscopic terms, the Δ-ruler is applied to each 128-dimensional head before its turn-chart action.

### 5.2 Normal RoPE pairing

Qwen3 uses consecutive coordinate pairs:

`(x_0,x_1), (x_2,x_3), ..., (x_126,x_127)`

It does not use the NeoX half-split pairing `(x_i,x_{i+64})`. This matters theoretically because a rotation chart is defined by both its angle and its planes. A correct turn tick applied to the wrong coordinate planes is a different representation of the phase group.

Thus native RoPE requires two identifications:

1. Continuous pretrained phase is represented on `T_256^(turn)`.
2. The finite turn acts on the same consecutive two-planes used during training.

The fractional aperture chart and the turn chart must remain distinct. `5/256` represents Δ as a ratio. It is not a rotary angle.

### 5.3 Grouped-query memory

Thirty-two query heads share eight K/V heads. Each key head therefore serves four query heads. Rotary transport belongs to the key head itself and must occur once per key head, not once per query-to-key association.

This reveals a useful separation. GQA is an incidence structure between query fibers and memory fibers. It does not multiply the physical memory state. Native chirality and shell data should be attached to each actual key memory head, while query heads form transport classes against that shared state.

### 5.4 Final-row selection is causal topology

Qwen3 selects requested output rows inside the final transformer block before final normalization and output projection. This selection is not a harmless presentation detail. It determines which residual states survive from the block trajectory into the model tail.

The theoretical lesson is general: graph operations that select, copy, or reshape a load-bearing state can belong to the semantic topology even when they contain no learned parameters. Native ownership must follow data provenance, not only operation names. A selection whose source is a native-owned block is itself part of that ownership boundary.

### 5.5 Final norm and output projection form a distinct tail

Qwen3 exposes a final residual, applies final RMS normalization, and then applies `output.weight` to obtain logits. The native block trajectory can be complete while this tail remains externally evaluated.

This gives three nested notions of ownership:

1. **Block ownership:** all 36 recurrent transformer blocks are native.
2. **Whole-model forward ownership:** embedding entry, blocks, final norm, and logits are native.
3. **End-to-end generation ownership:** forward map plus token selection are native or explicitly declared chassis boundaries.

These notions should not be conflated. The current theoretical program ultimately asks for the second, with an explicit decision about the third.

---

## 6. Causality and layer-major prefill

A causal transformer layer maps all prompt residuals at layer ℓ into K/V memory and updated residuals before layer ℓ+1 acts. Therefore the partial order is:

`(t,ℓ) precedes (t,ℓ+1)`

and, for attention memory at fixed ℓ:

`(j,ℓ) is available to (t,ℓ) whenever j ≤ t`

A layer-major schedule realizes this order:

```text
for ℓ in layers:
    for t in prompt positions:
        evaluate block(t, ℓ)
```

A token-major schedule violates the intended factorization because token `t` can reach layer `ℓ+1` before later prompt tokens have established their layer-ℓ memory. Even if each local operation appears causal, the global prompt state is not the pretrained layer map.

Genealogy depth remains `t·36+ℓ` even though physical execution is layer-major. Genealogy labels semantic position in the model trajectory. Scheduler order is an implementation of dependency constraints, not a redefinition of causal time.

Absolute model positions are similarly authoritative. Position determines RoPE phase, KV address, and Genealogy depth. Call count is not a substitute because batching, request reset, and decode scheduling can change calls without changing semantic position.

---

## 7. Native laws at transformer sites

### 7.1 Weight MatMul

**Demand:** preserve learned Q1_0 amplitude while realizing manifold routing.

**Law:** exact Q1_0 by q8 controller contraction times manifold gain.

**Reason:** a Q1_0 weight is already a finite bipolar codebook with a scale. The signs define discrete support and the scale defines amplitude. The ledger compiles that support into shell routing without discarding the learned controller magnitude.

### 7.2 Key memory

**Demand:** preserve causal QK magnitude and the key's transport coordinate.

**Law:** Q8_0 key memory plus matching `k_chi6`.

**Reason:** QK energy and transport class are different observables. The score requires live magnitude geometry. Chirality records the finite class used by lift, shell, and Genealogy. A χ-only score would collapse the controller.

### 7.3 Value memory

**Demand:** preserve the individuality carried by value content.

**Law:** Q8_0 value memory and native value reduction.

**Reason:** Δ greater than zero requires distinguishable excitations to survive transport. Value vectors carry learned content, so shell coordinates cannot replace their decoded amplitude. A finite magnitude chart with a faithful decoder preserves both native storage and content.

### 7.4 Attention weighting

For query chirality `χ_Q` and key chirality `χ_Ki`, define:

`q_i = χ_Q ⊕ χ_Ki`

and shell distance:

`N_i = popcount(q_i)`

Native attention uses polynomial shell weight `λ^N` and algebraic selection rather than `exp(score)`. The full law joins two inputs:

- controller QK magnitude, which preserves the learned energy ordering
- manifold transport class, which supplies shell occupation and Genealogy coupling

This joint law avoids two category errors. It does not retain stock Softmax as the architecture, and it does not replace learned QK magnitude with chirality alone. In spectral form, chirality is diagonalized by WHT and shells by Krawtchouk structure.

### 7.5 RoPE

RoPE is phase transport. Its native chart is `T_256^(turn)`, tied to absolute token position and compatible with K4 Genealogy phase. For Bonsai, the finite action must retain the Qwen3 consecutive-pair planes, grouped-query incidence, frequency base `10^6`, and YaRN scale `1/4`.

This is a chart compilation, not an aperture perturbation of stock angles. Δ governs fractional gain defect. Turn ticks govern rotation.

### 7.6 Normalization

The Δ-ruler represents positive gain by:

`n = round(log2(g/g0) / Δ)`

`ĝ = g0 · 2^(n·Δ)`

Two reference moments occur in Qwen3 normalization and must remain distinct:

- the inverse RMS gain is dimensionless and centered around reference 1
- the learned norm vector is a tensor-local operator and is centered around its geometric mean

Using one baseline for both collapses state normalization and learned calibration into one coordinate.

Finite exactness also imposes a word-size condition. At width 4096, direct fixed-point squaring can overflow a signed 64-bit accumulator for valid late-layer residuals. A lawful finite moment therefore normalizes by `amax`, accumulates bounded Q15 squares, and restores the outer scale. This is not a fallback to Euclidean semantics. It is the requirement that a finite chart be closed over the actual dynamic range of the model.

### 7.7 FFN and SwiGLU

The three FFN matrices obey the same ledger MatMul law. The nonlinear gate is expressed through shell occupation and K4 family rather than sigmoid or SiLU as a transcendental decision law.

The shell is the radial nonlinear coordinate forced by Ω. Family supplies phase selection. Their product gives a finite gate indexed by occupation and Genealogy phase. A LUT of stock SiLU can serve as a diagnostic chart, but it is not the native law because it preserves the external nonlinearity rather than deriving selection from the carrier.

The operator decomposition:

`W·x = P_Q(W)·x + D_Q(W)·x`

describes linear quotient defect. It must not be confused with the FFN pointwise gate or with residual-stream mixing.

### 7.8 Residual stream

The residual mixer is:

`x ← x + y · (1 + Δ·m)`

where:

`m = (N - 3)/3`

The running state carries identity. The branch update carries individual excitation. Aperture times polarization is therefore the natural coupling between them. It permits the update to vary with the carrier's horizon imbalance while preserving the common stream.

This residual law is distinct from `P_Q + D_Q`, which decomposes a linear operator rather than mixing two stream branches.

### 7.9 Common Source projection

External embeddings are not automatically kernel states. A declared projection Pi is required to derive `(u6,v6)` from embedding signs and support structure. Until such a projection is applied, `GENE_MAC_REST` is the universal reference.

`Pi_basis` preserves basis-level information for operator analysis. `Pi_summary` provides chirality, shell, and resonance summaries for request grouping. Polar encoding can instantiate a summary projection, but it does not prove that an F32 embedding row is itself an element of Ω.

---

## 8. Float freedom and semantic ownership

Native semantics and float-free implementation are related but distinct claims.

A site is semantically native when its decision law is expressed by the datatype, Genealogy, finite charts, and controller exactness rather than by a stock transformer primitive. A runtime is operationally float-free only when its storage and arithmetic also avoid floating-point execution in the hot path.

A tick-based RoPE law applied to F32 row buffers is semantically on the native turn chart but not operationally float-free. A Q1_0 by q8 integer dot whose decoded scales and outputs are held in F32 has an exact integer controller core but still uses a floating interoperability chart. The same distinction applies to residual buffers, normalization gains, and attention accumulation.

This distinction prevents two opposite mistakes:

- treating zero stock-operation counters as proof of float-free algebra
- treating temporary F32 coordinates as proof that no native semantic law exists

The final architecture requires both semantic ownership and operational closure of the integer or dyadic chart.

---

## 9. Category errors to avoid

- Replacing live QK magnitude with χ-only or shell-only energy.
- Keeping `exp` Softmax and calling it native because K/V are quantized.
- Using percolation or λ as an unstructured substitute for learned attention energy.
- Conflating `T_256^(frac)` with `T_256^(turn)`.
- Applying the correct rotary angle to the wrong Qwen3 coordinate planes.
- Rotating shared GQA key state once per query head.
- Conflating the residual mixer `1 + Δ·m` with operator defect `P_Q + D_Q`.
- Treating a SiLU LUT as the final carrier-native nonlinearity.
- Treating embedding rows as anchors without a declared Pi.
- Resetting Genealogy per token or advancing it by function-call count.
- Identifying `HQVMLEDS` with runtime Genealogy.
- Ignoring parameter-free graph selection because it is not a MatMul.
- Claiming whole-model ownership when only transformer blocks are native.
- Claiming float-free execution from stock-op counters alone.

---

## 10. Conclusion

Gyroscopic inference compiles a frozen transformer into one finite causal trajectory. The manifold determines shell and transport class. The controller preserves learned magnitude through exact Q1_0 by q8 contraction. Genealogy supplies request origin, depth, and K4 phase. Aperture, occupation, polarization, and finite turn charts replace external transcendental decision laws with objects native to Ω.

Bonsai and Qwen3 taught an important refinement of this program. Native compilation must preserve the pretrained model's causal topology: per-head Q/K normalization, consecutive rotary planes, grouped-query memory incidence, layer-major prompt dependencies, absolute positions, final-row selection, and the distinct final norm and logits tail. These are the boundary conditions under which the learned operator exists.

The resulting architecture is not a stock transformer plus gyroscopic annotations. It is a gyroscopic trajectory whose controller was learned in transformer form and whose topology is compiled faithfully into the native datatype. Full completion requires extending that ownership through model entry and tail, then eliminating the remaining floating interoperability charts so that semantic and operational native exactness coincide.
