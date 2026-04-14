# GyroLabe
## Specification

GyroLabe is the native execution and lowering substrate of the Gyroscopic ASI stack. It provides hardware-near realization of the exact transforms, structured operator analysis, and hybrid application surfaces defined by the QuBEC transform algebra. Every operation GyroLabe performs is either an exact realization of a native QuBEC chart or an explicit routing decision between exact and residual execution paths.

**Versioning note:** This specification is a living document. Changes to enum encodings, normalization contracts, or projection classes must include migration notes and conformance test updates.

---

## 1. Role in the Stack

GyroLabe sits between the aQPU kernel reference semantics and the systems that consume computational results.

Below it is the kernel law: the 24-bit state model, the 4096-state Ω manifold, the chirality register, the exact operator algebra, and the reference transform surfaces defined in `src`. Above it are GyroGraph, bridge logic, and application control surfaces.

GyroLabe has five responsibilities.

**Exact kernel execution.** Stepping, q-extraction, chirality distance, signature accumulation, Ω scans, shell histograms, and all kernel observables. These operations are bitwise identical to the reference semantics under any conditions.

**Native transform execution.** The 64-point Walsh-Hadamard transform on GF(2)⁶, the 7-point Krawtchouk transform on shell space, and the 4-point K4 character transform on family space. These are the unique exact harmonic bases of the three native climate charts. GyroLabe provides hardware-near execution of all three.

**Structured operator analysis.** For any 64-wide operator block, GyroLabe determines which exact quotient class it belongs to, computes the projection into that class, and produces the residual defect. This is the structural analysis step that precedes all lowering decisions.

**64-wide lowering and hybrid application.** External tensors of arbitrary width tile into 64-wide blocks. Every block is applied by the exact decomposition `W · x = P_Q(W) · x + D_Q(W) · x`. Native quotient transforms realize `P_Q`; the defect `D_Q` remains part of the same semantics and is not a correctness fallback.

**Packed arithmetic execution.** K4 lattice matrix contractions, packed GEMV using the signed-support arithmetic surface, and repeated-application paths for cached operator blocks.

GyroLabe is execution-focused. It realizes the transforms defined in the QuBEC Transform Algebra document with conformance obligations against the reference semantics defined in the SDK.

---

## 2. Exactness Model

GyroLabe operates in three precisely defined exactness classes. Every operation belongs to exactly one class. Mixed-class compositions are explicit and tracked.

### 2.1 Integer exact

Operations that produce bitwise-identical results to the reference semantics under integer arithmetic with no approximation:

- byte stepping and state transitions
- q-class, family, and micro-reference extraction
- chirality word extraction and chirality distance
- signature accumulation and application
- unnormalized Walsh-Hadamard transform
- shell index, shell histogram, Ω scans
- parity commitments
- K4 D₁₁ popcount contractions in the spinorial regime
- all horizon and shell observables

These operations must match the `src` reference surfaces exactly. No platform-dependent variation is permitted.

### 2.2 Dyadic exact

Operations that are exact as rational values with denominators that are powers of two or small combinatorial factors:

- normalized WHT (divide by 64, realized as right shift by 6)
- K4 character normalization (divide by 4, realized as right shift by 2)
- shell probability from shell histogram
- Krawtchouk-normalized transform coefficients
- climate occupation fractions
- dyadic-exact operator projections

These operations remain exact and do not require floating-point semantics. Internally they are tracked as:

```
(numerator: int64, exponent: int8)
```

representing numerator · 2^{−exponent}. Exponents accumulate additively under composition. Integer-exact sources enter dyadic representation with exponent 0.

### 2.3 Numerically faithful residual

Operations outside the exact quotient structure:

- non-native dense operator residuals
- external float tensor interfaces
- approximate hardware backends
- quantized inference engine integrations

This class is not a failure mode. It is the explicit execution path for the defect component D_Q(W) in hybrid application. Results in this class are produced to the precision of the underlying backend and are not required to match an integer or dyadic reference.

Hybrid operations that combine classes 2.1 or 2.2 with class 2.3 must track the boundary explicitly. The exact component and the residual component are summed only at the output boundary.

---

## 3. Execution Surfaces

GyroLabe exposes six cooperating surfaces.

### 3.1 Exact kernel surface

The C implementation of the kernel-facing algebra. All operations are integer exact and produce bit-identical results across platforms.

This surface covers:

- signature accumulation over byte ledgers
- compiled signature application to rest and arbitrary states
- byte decomposition: q-class, family, micro-reference
- chirality word extraction and chirality distance, pairwise and adjacent
- Ω stepping: single-byte and batch
- shell histogram computation over state sequences
- exact q-sector selection with optional shell weighting
- state scan from arbitrary start state

### 3.2 Native transform surface

Hardware-near execution of the three native spectral transforms.

**Walsh-Hadamard transform (WHT64).** The 64-point Walsh-Hadamard transform on the chirality register GF(2)⁶. Executes via butterfly stages of (x, y) → (x + y, x − y). Requires 384 additions and subtractions. No multiplications. Self-inverse up to a factor of 64, realized as a right shift by 6 in the dyadic-exact path.

Single transform:

```
Input:  data   int32[64], unnormalized chirality space
Output: data   int32[64], unnormalized spectral space (in-place)
```

Batch transform:

```
Input:  data_batch   int32[B, 64]
Output: data_batch   int32[B, 64]  (in-place, all B cells)
```

The butterfly structure parallelizes naturally: each of 6 stages processes 32 independent pairs. With AVX2, a stage processes 8 pairs per vector operation, giving 4 vector operations per stage and 24 total.

The int32 WHT64 path is integer exact provided all intermediate butterfly sums remain in int32 range. If overflow is possible, execution must promote to an int64 scratch or equivalent widened path while preserving exact arithmetic.

**Krawtchouk transform (Krawtchouk7).** The 7-point Krawtchouk transform on shell space, using the exact Krawtchouk polynomial coefficients from the SDK reference surface.

```
Input:  shell_hist   int32[7]
Output: spectral     int32[7]  (Krawtchouk spectral coefficients)
```

Inverse transform available as Krawtchouk7_inverse.

The unnormalized Krawtchouk7 transform is integer exact on integer shell histograms. Normalized coefficients are dyadic-exact or tracked as exact rational values depending on representation choice. The inverse transform preserves the same exactness contract.

**K4 character transform (K4Char4).** The 4-point Hadamard transform on the family histogram.

```
Input:  family_hist   int32[4]
Output: character     int32[4]  (K4 character decomposition)
```

### 3.3 Structured operator surface

All 64-wide operator analysis is defined on the canonical lexicographic ordering of GF(2)⁶. Quotient tests, Walsh diagonalization, shell partitioning, and χ × gauge projections are evaluated relative to this basis. Any external block must be mapped into this basis before structure analysis is performed.

For any 64-wide operator block W, GyroLabe determines which exact quotient class it belongs to, computes the projection, and produces the residual defect.

**Quotient classes supported:**

| Class | Parameters | Test | Native path |
|---|---|---|---|
| Shell-radial | 7 | Shell variance across rows | Krawtchouk diagonal |
| Shell × gauge | 28 | Shell × K4 sector | Krawtchouk ⊗ K4Char diagonal |
| χ translation-invariant | 64 | Circulant structure | WHT diagonal |
| χ × gauge | 256 | WHT ⊗ K4Char structure | WHT ⊗ K4Char diagonal |
| Generic | 4096 | None | K4 defect evaluation |

**Encoding warning.** The C enum `gyro_class_id_t` in `src/tools/gyroscopic/gyrolabe_registry.h` and the bridge enum `gyrolabe_operator_class_t` in `src/tools/gyroscopic/gyrograph_types.h` use different numeric encodings. Do not cast between them by integer value. Use named constants or an explicit mapping table.

**Structure analysis algorithm:**

```
Input:  W         operator block, int32[64, 64] or float[64, 64]
        threshold real, default 0.01

Output: report
          class         string
          method        string
          eigenvalues   int32 or float array
          scr           real (structure capture ratio)
          defect_norm   real

1. Test translation-invariance:
       W_circ ← circulant reconstruction from W[0, :]
       if ‖W − W_circ‖_F / ‖W‖_F < threshold:
           φ ← WHT of first row of W
           return {class: "chi-invariant",
                   method: "wht-diagonal",
                   eigenvalues: φ,
                   scr: 1 − ‖W − reconstruct(φ)‖_F / ‖W‖_F}

2. Test shell-radial structure:
       For each shell r, compute mean of W[i, j] over (i, j)
           where popcount(i) = r and popcount(j) = r
       shell_var ← variance of off-diagonal shell means
       if shell_var < threshold:
           λ ← 7 mean values per shell
           return {class: "shell-radial",
                   method: "krawtchouk-diagonal",
                   eigenvalues: λ,
                   scr: ...}

3. Test χ × gauge structure:
       T ← WHT ⊗ K4Char basis transform of W
       diag_frac ← diagonal norm of T / total norm of T
       if diag_frac > 1 − threshold:
           return {class: "chi-x-gauge",
                   method: "wht-k4char-diagonal",
                   eigenvalues: diagonal of T,
                   scr: diag_frac}

4. return {class: "generic", method: "k4-defect-eval", scr: ‖P_Q(W)‖_F / ‖W‖_F}
```

Analysis is performed once per operator block and the report is cached. Subsequent applications use the cached report.

**Structure capture ratio:**

```
SCR_Q(W) = ‖P_Q(W)‖_F / ‖W‖_F
```

Values above 0.9 indicate the native path captures most of the operator. Values below 0.2 indicate the operator has little structure in this quotient class.

**Projection and defect:**

```
P_Q(W) = projection of W into quotient class Q
D_Q(W) = W − P_Q(W)
```

Both are available as explicit outputs for hybrid execution.

### 3.4 64-wide lowering and hybrid application surface

This surface tiles external tensors, routes computation, and combines exact and residual results.

**Block tiling:**

```
Input:  W     dense matrix, shape (rows, d)

Output: block_registry

1. pad d to next multiple of 64 if d mod 64 ≠ 0
2. for each block b in range(d // 64):
       W_block ← W[:, b·64 : (b+1)·64]
       report  ← analyze_operator(W_block)
       block_registry[b] ← (W_block, report)

return block_registry
```

**Hybrid block application:**

```
Input:  block_registry
        x   input vector, width d

Output: result vector

for each block b in block_registry:
    x_block ← x[b·64 : (b+1)·64]
    report  ← block_registry[b].report

    if report.class == "chi-invariant":
        y_b ← WHT64(x_block) * report.eigenvalues   (pointwise)
        y_b ← WHT64(y_b) >> 6

    elif report.class == "shell-radial":
        s   ← shell_histogram(x_block)
        y_b ← Krawtchouk7_inverse(
                   Krawtchouk7(s) * report.eigenvalues)

    elif report.class == "chi-x-gauge":
        y_b ← apply_wht_k4char(x_block, report.eigenvalues_256)

    else:
        y_b ← apply_k4_defect_eval(W_block, x_block)

return concatenate(y_b for all b)
```

The `>> 6` normalization denotes division by 64. Integer-exact paths use right shift, while float paths use multiplication by `1.0f / 64.0f`. Both represent the same dyadic normalization contract.

When the operator block is split into structured and residual components, both are applied and summed:

```
y_b = P_Q(W_block) · x_block   (native exact path)
    + D_Q(W_block) · x_block   (K4 defect path)
```

This decomposition is the semantic definition of block application. `SCR_Q(W_block)` is a workload descriptor, not a correctness gate.

**Exact substitution law.** When `SCR_Q(W_block) = 1.0`, then `D_Q(W_block) = 0`, so the same decomposition collapses to the native branch alone. This is a degenerate cost case, not a separate correctness regime.

**Hybrid preservation law.** For any W_block and quotient class Q:

```
W_block · x = P_Q(W_block) · x + D_Q(W_block) · x
```

If the native path computes P_Q(W_block) · x exactly or dyadically-exactly, and the dense backend computes D_Q(W_block) · x to its native precision, the hybrid result equals the full operator result to the precision of the dense backend. No quality is lost relative to full dense execution.

### 3.5 Packed arithmetic surface

Matrix computation through K4 lattice decomposition.

The K4 lattice matrix decomposes integer dot products into four sectors:

```
M(q, k) = [[D₀₀, D₀₁],
            [D₁₀, D₁₁]]

⟨q, k⟩ = D₀₀ + B·(D₀₁ + D₁₀) + B²·D₁₁
```

where B = 2¹⁶.

**Spinorial regime.** When H ∈ {−1, 0, +1}ⁿ for both vectors, D₁₁ is computed via boolean support intersection:

```
D₁₁ = popcount(q⁺ ∧ k⁺) + popcount(q⁻ ∧ k⁻)
     − popcount(q⁺ ∧ k⁻) − popcount(q⁻ ∧ k⁺)
```

This is integer exact.

**GEMV with packed matrices.** For repeated application of the same operator to many input vectors, the operator is packed once and the packed form is reused:

```
pack_matrix64(W):
    for each row r:
        W_sign[r]    ← uint64 sign mask (bit j = sign bit of W[r, j])
        W_bp[r, k]   ← uint64 for each magnitude bit plane k
    store scale_w
    return packed_matrix

apply_packed64(packed_matrix, x):
    scale_x ← per-vector scale derived from x
    result  ← K4 lattice accumulation using W_bp and sign masks
    recover ← result * scale_w * scale_x
    return recover
```

Packing is performed once per operator. Repeated GEMV uses the packed form without repacking.

**Regime detection.** The regime (carrier, spinorial, dense) is determined from the data at execution time. No separate API selection is required.

### 3.6 Bridge and vendor integration surface

GyroLabe exposes integration hooks for external runtimes and inference frameworks.

The primary validated integration is llama.cpp via `ggml-gyroscopic`. The relevant sources are:

- `gyrolabe_wht_matmul.c`: wht-based matmul routing
- `gyrolabe_pack.c`: packed matrix preparation
- `gyrolabe_core.c`: kernel-facing exact operations
- `gyrolabe_wht.c`: fast WHT implementation
- `gyrolabe.h`, `gyrolabe_simd.h`: headers and SIMD abstractions

The wht path is the active matmul routing path for the llama.cpp integration. When wht conditions are not met for a given operator block, the integration falls back to the standard dense path.

Other runtime integrations follow the same pattern: the hybrid lowering surface of Section 3.4 is the primary interface. The bridge or vendor hook maps external weight tensors and activations into 64-wide blocks, routes through the structured analysis and hybrid application pipeline, and returns results in the format expected by the external runtime.

---

## 4. Conformance Obligations

A conforming GyroLabe implementation satisfies the following obligations.

### 4.1 Integer-exact conformance

All integer-exact operations must produce bitwise-identical results to the corresponding reference surfaces in `src/api.py`, `src/constants.py`, and `src/kernel.py`. This includes all kernel stepping, q-extraction, chirality distance, signature accumulation, and shell and horizon observables.

### 4.2 Dyadic-exact conformance

All dyadic-exact operations must produce results that equal the exact rational values to the represented precision. For WHT normalization this means the integer result after right-shifting by 6 must equal the result of the reference normalized WHT rounded to the same integer.

### 4.3 Transform conformance

The unnormalized WHT64 must satisfy:

```
WHT64(WHT64(f)) = 64 · f   for all f : GF(2)⁶ → ℤ
```

This must hold exactly in integer arithmetic.

The Krawtchouk7 transform must match the exact coefficients from `src/api.py::KRAWTCHOUK_7`.

The K4Char4 transform must match the character table of K4.

### 4.4 Hybrid conformance

The hybrid exact-residual path must satisfy:

```
apply_hybrid(W, x) = apply_native(P_Q(W), x) + apply_dense(D_Q(W), x)
```

The native and dense components must be computed and summed without interaction. The residual component must not contaminate the exact component.

### 4.5 Structure analysis conformance

For any operator `W` and chosen quotient class `Q`, implementations must evaluate `P_Q(W) · x + D_Q(W) · x` and match the reference semantics to the stated numeric precision. When `SCR_Q(W) = 1.0`, this obligation is unchanged and `D_Q(W)` is identically zero by value.

#### 4.6 Conformance summary

| Surface | Exactness class | Reference source |
|---|---|---|
| Byte stepping | Integer exact | SDK / src |
| Signature accumulation and application | Integer exact | SDK / src |
| q-class, family, micro-reference extraction | Integer exact | SDK / src |
| WHT64 unnormalized | Integer exact | Transform semantics |
| WHT64 normalized | Dyadic exact | Transform normalization contract |
| Krawtchouk7 unnormalized | Integer exact | SDK coefficient table |
| Krawtchouk7 normalized | Dyadic or exact rational | SDK coefficient table |
| K4Char4 | Integer or dyadic exact | K4 character table |
| Hybrid block application | Mixed | Structured plus residual contract |

---

## 5. Performance

GyroLabe's performance significance lies in the structural cost reduction that native chart selection achieves. The figures below are symbolic operation counts, not measured wall-time benchmarks.

### 5.1 Native transform costs

| Operation | Additions/subtractions | Multiplications |
|---|---|---|
| WHT64 (unnormalized) | 384 | 0 |
| WHT64 + pointwise + inverse | 832 | 64 |
| Krawtchouk7 | 42 multiply-add | 42 |
| K4Char4 | 12 | 0 |

### 5.2 Single-step structured vs dense application

For a 64-wide translation-invariant operator applied to a 64-element vector:

**Dense:**
```
64² = 4,096 multiply-accumulates
Memory: 64 · 64 · 4 bytes = 16,384 bytes
```

**Native WHT path:**
```
384 + 64 + 384 = 832 arithmetic operations
Memory: 2 · 64 · 4 = 512 bytes
```

Symbolic operation ratio: approximately 5×.
Symbolic memory ratio: 32×.

### 5.3 n-step evolution

For n applications of a translation-invariant ensemble using the spectral composition algorithm:

**Dense method:**
```
log₂(n) matrix multiplies, each 64³ = 262,144 multiply-accumulates
```

**Native spectral method:**
```
768 add/sub  (two WHTs)
64 · log₂(n) multiplies  (elementwise exponentiation by squaring)
64 multiplies  (pointwise composition)
```

Symbolic operation ratio at n = 1000: approximately 1,800×.

The ratio grows with n because the dense method exponentiates 64×64 matrices while the spectral method exponentiates 64 scalars. Both use binary exponentiation.

### 5.4 Batch execution

For B cells with a shared ensemble:

The ensemble WHT is computed once. The per-cell forward and inverse WHTs parallelize across cells. With AVX2, the 64-element butterfly structure maps to 8 cells processed in parallel per vector width.

Symbolic operation ratio at B = 32: approximately 5× fewer arithmetic operations than B independent dense applications.

---

## 6. Technical Notes

The native C library requires any C11 compiler and builds automatically on first import. The Python surface uses ctypes and numpy bindings. The packed arithmetic path is forward-only. The C library uses process-global tables initialized once per process and is safe for single-threaded and OpenMP-parallel use. Initialization is idempotent and may be called at startup.

The Python binding surface in `ops.py` wraps native operations, manages automatic native builds across GCC, Clang, and MSVC, and exposes packed matrix and vector objects for repeated application.

---

## 7. Relation to Other Stack Components

**aQPU Kernel SDK.** The SDK defines the normative reference semantics for all exact operations. GyroLabe realizes those semantics at hardware speed. Conformance is defined by parity against the SDK reference surfaces.

**QuBEC Transform Algebra.** The transform algebra document defines the mathematical basis for all four native transforms and their equivalence structure. GyroLabe is the execution realization of that algebra. The quotient classes, structure analysis, and hybrid application protocols in GyroLabe correspond directly to the structures defined in that document.

**GyroGraph.** GyroGraph consumes GyroLabe surfaces for cell stepping, rolling memory updates, spectral evolution, structured operator analysis, and hybrid block application. GyroGraph does not reimplement any of these; it calls GyroLabe native operations directly. The SLCP spectral64 field in GyroGraph is the normalized WHT64 of chi_hist64, computed through the GyroLabe native transform surface.

**External runtimes.** The bridge and vendor integration surface connects GyroLabe to external inference frameworks. The hybrid lowering surface is the primary integration interface: external weight tensors tile into 64-wide blocks, route through structure analysis and hybrid application, and return results in the format expected by the external runtime.