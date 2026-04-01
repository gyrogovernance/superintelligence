

## Master instructions for File analysis task

You are analyzing files from `external/llama.cpp/ggml/src/ggml-gyroscopic/`
(our custom CPU backend copy) and selected test/tool files.

For each file, produce a document like `workflow/exports/1.md`.

## Rules

1. **Do not summarize the whole file.** Extract only what the playbook
   categories ask for.

2. **Focus on what matters for gyroscopic matmul replacement.** If a
   function has nothing to do with MUL_MAT, MUL_MAT_ID, OUT_PROD,
   SOFT_MAX, or the dispatch path that reaches them, note its existence
   in one line and move on.

3. **For SIMD/kernel code:** extract the exact instruction sequence pattern
   (load, widen, madd, horizontal sum, scale), not prose descriptions.
   Use pseudocode or intrinsics names. Note block sizes, unroll counts,
   and register reuse patterns.

4. **For dispatch code:** trace the call chain from the op enum to the
   actual function that runs. Name every branch point, every dtype check,
   every "if repack then X else Y" fork. These are bypass risks.

5. **Mark bypass risks explicitly.** If a code path can reach matmul
   execution without going through our hook points (vec.cpp vec_dot_f32,
   arch/x86/quants.c vec_dot_q8_0_q8_0, ggml-cpu.c direct GEMM), say so.

6. **Note the 32-vs-64 grain.** When you see block sizes, tile sizes, or
   loop strides, note whether they are 32-wide (Q8_0 native) or could
   naturally fuse to 64-wide.

7. **Do not analyze ARM, AMX, KleidiAI, SpaceMIT, or llamafile paths.**
   x86 AVX2 only for now.

8. **Each document must end with a "Gyroscopic action" section** listing
   concretely:
   - What hook/redirect is needed (if any)
   - What kernel grammar to adopt into codec.c (if any)
   - What bypass risk exists (if any)
   - What to record in log.md (if any)

9. **Exactness class:** When describing a computation path, label it:
   - kernel-exact (pure integer, no float involvement)
   - deterministic-numeric (integer core + float scale/accumulation)
   - float (full floating point)
   This matters for honest claims.

10. **File size guidance:**
    - Under 500 lines: full relevant extraction
    - 500-1500 lines: focus on MUL_MAT-relevant functions only
    - Over 1500 lines: index all functions first (one line each),
      then deep-extract only MUL_MAT-relevant paths
```

---

## Per-file briefs

### Export 2: `ggml-gyroscopic/ops.cpp`

```markdown
# Export 2: `ggml-gyroscopic/ops.cpp`

## What to extract

This is the op-level dispatch front door.

1. Find the main dispatch function (likely `ggml_compute_forward` or similar).
   List every `GGML_OP_*` case that appears.

2. For these specific ops, extract the FULL dispatch path (what function
   gets called, what type checks happen, what branches exist):
   - GGML_OP_MUL_MAT
   - GGML_OP_MUL_MAT_ID
   - GGML_OP_OUT_PROD
   - GGML_OP_SOFT_MAX
   - GGML_OP_RMS_NORM
   - GGML_OP_ROPE (just note if present and what it calls)
   - GGML_OP_UNARY (note which unary sub-ops: GELU, SILU, etc.)

3. For MUL_MAT specifically: does this file handle the actual computation,
   or does it delegate to ggml-cpu.c? Trace the delegation.

4. Note any existing `#ifdef GGML_USE_GYROSCOPIC` blocks.

5. List ALL ops that appear in the dispatch, even ones we don't care about,
   as a one-line inventory (op name only, no detail).

## What NOT to extract
- Implementation details of ops we keep stock (reshape, view, concat, etc.)
- Comments about the file's history or style
```

### Export 3: `ggml-gyroscopic/ggml-cpu.c`

```markdown
# Export 3: `ggml-gyroscopic/ggml-cpu.c`

## What to extract

This is the orchestration and matmul execution core.

1. **MUL_MAT path (CRITICAL):** Find `ggml_compute_forward_mul_mat` or
   equivalent. Extract:
   - How it decides between vec_dot path vs GEMM path vs repack path
   - The chunking logic (ith/nth, row ranges)
   - Where vec_dot function pointers come from (type_traits?)
   - Where workspace (wdata) is allocated and sized
   - Any existing gyroscopic hooks (#ifdef GGML_USE_GYROSCOPIC)
   - The exact branch that routes to our bridge GEMM call

2. **type_traits / vec_dot routing:** Find where vec_dot function pointers
   are registered per dtype pair. This is how Q8_0 x Q8_0 reaches the
   actual dot product function.

3. **Repack decision:** Find where the code decides to use repack-based
   forward_mul_mat vs generic vec_dot loop. This is a BYPASS RISK.
   Extract the condition exactly.

4. **OUT_PROD path:** Find the outer product implementation. Note its
   structure.

5. **Thread barrier / sync patterns:** Just note the pattern (barrier
   between phases?), don't deep-analyze.

6. **Ignore:** Non-matmul ops, backend registration boilerplate, memory
   management functions.

## Expected output
- A clear call-chain from "MUL_MAT op arrives" to "dot product function
  is called" with every branch point named.
- The repack bypass condition stated explicitly.
- Current gyroscopic hook locations with their conditions.
```

### Export 4: `ggml-gyroscopic/ggml-cpu.cpp`

```markdown
# Export 4: `ggml-gyroscopic/ggml-cpu.cpp`

## What to extract

This is typically the C++ wrapper / backend registration side.

1. `supports_op`: Find the function that declares which ops this backend
   supports. Extract the conditions for MUL_MAT, MUL_MAT_ID, OUT_PROD.

2. `extra_buffer_type`: Find what extra buffer types are used and when.
   This matters because repack paths often depend on extra buffer
   availability.

3. Any backend initialization that affects matmul routing.

4. **Keep brief.** This file is mostly plumbing. Only detail what affects
   the matmul dispatch decision.
```

### Export 5: `ggml-gyroscopic/arch/x86/quants.c`

```markdown
# Export 5: `ggml-gyroscopic/arch/x86/quants.c`

## What to extract

This is the PRIMARY optimization reference for codec.c.

1. **Find `ggml_vec_dot_q8_0_q8_0`.** Extract:
   - The EXACT intrinsics sequence (load, widen/madd, accumulate, reduce)
   - Block loop structure (how many blocks per iteration, unrolling)
   - Horizontal sum pattern (register-based or store-based?)
   - Scale handling (when/where fp16->float conversion happens)
   - Whether it uses _mm256_maddubs_epi16 or the widen-then-madd approach

2. **Find any other q8_0 functions** (gemv, gemm variants if present).
   Note their signatures and key differences from the dot product.

3. **Find our existing hook** (#ifdef GGML_USE_GYROSCOPIC in this file).
   What does it currently intercept? What conditions gate it?

4. **Note the block struct layout** (where is it defined — here or in
   a shared header?).

5. **32-vs-64 grain:** Note the natural processing width. How many int8
   values are processed per AVX2 instruction? Is there any 64-wide
   loop structure?

6. **Exactness class of the stock path:** The integer dot product core
   is exact. The scale multiply is float. Label the boundary.

## What NOT to extract
- Non-Q8 quantization formats (Q4, Q5, etc.) unless they share
  infrastructure with Q8
- ARM/NEON paths
- Anything outside x86 AVX2
```

### Export 6: `ggml-gyroscopic/quants.c` (top-level)

```markdown
# Export 6: `ggml-gyroscopic/quants.c`

## What to extract

This is the generic/dispatch layer above arch-specific quants.

1. How does `ggml_vec_dot_q8_0_q8_0` dispatch between architectures?
   Is there a function pointer table? Direct call? Preprocessor?

2. Is there a reference (non-SIMD) implementation of the Q8_0 dot
   product here?

3. Block type definitions: is `block_q8_0` defined here or in a header?
   Extract the exact struct layout.

4. Any quantization/dequantization functions for Q8_0 that matter for
   understanding data flow into matmul.

5. **Keep focused on Q8_0.** Other quant formats are not our current target.
```

### Export 7: `ggml-gyroscopic/simd-gemm.h`

```markdown
# Export 7: `ggml-gyroscopic/simd-gemm.h`

## What to extract

This is the GEMM tiling reference.

1. **Tile sizes:** What are M_tile, N_tile, K_tile? Are they compile-time
   constants or runtime parameters?

2. **Microkernel interface:** What does a single tile computation look like?
   How many accumulator registers? What is the inner loop structure?

3. **How does this connect to MUL_MAT?** Is it called from ggml-cpu.c
   directly, from repack, or both?

4. **Register blocking strategy:** How many output elements are computed
   per inner loop iteration? This directly informs how to structure
   gyroscopic GEMM.

5. **Block size and Q8_0 compatibility:** Does this work with Q8_0 blocks?
   Or only with repacked/dequantized data?

## Expected output
- A sketch of the tiling strategy (M/N/K loop order, tile sizes)
- The microkernel register layout
- Whether this is the path that bypasses vec_dot
```

### Export 8: `ggml-gyroscopic/repack.cpp`

```markdown
# Export 8: `ggml-gyroscopic/repack.cpp`

## What to extract

This is a BYPASS RISK file.

1. **Find `forward_mul_mat` or equivalent.** This is the repack-based
   matmul path that can bypass the generic vec_dot loop entirely.
   Extract:
   - When does this path activate? (What tensor traits / buffer types
     trigger it?)
   - What GEMM/GEMV function does it call?
   - Does our gyroscopic hook cover this path?

2. **Repack layout transforms:** What does repacking do to Q8_0 data?
   How is it rearranged? This matters for understanding whether our
   hooks see the original or repacked layout.

3. **tensor_traits registration:** How does a tensor get marked as
   "use repack path"?

4. **Key question:** If repack is active for Q8_0, does the matmul
   EVER go through the generic vec_dot path? Or does repack always
   win? This determines whether hooking vec_dot alone is sufficient.

## Expected output
- Clear statement: "When repack is active, matmul for Q8_0 goes through
  [this path], which [does/does not] hit our hooks."
- The repack activation condition.
```

### Export 9: `ggml-gyroscopic/arch/x86/repack.cpp`

```markdown
# Export 9: `ggml-gyroscopic/arch/x86/repack.cpp`

## What to extract

1. x86-specific repack layout transforms for Q8_0 (if present).
2. x86-specific GEMM/GEMV implementations called from the repack path.
3. Intrinsics patterns used in repack GEMM (compare with quants.c
   patterns — are they the same or different microkernel grammar?).
4. Tile sizes and blocking specific to x86 repack GEMM.

## Focus
- Only Q8_0-relevant paths
- Only AVX2 paths
- Note any 64-wide or wider processing
```

### Export 10: `ggml-gyroscopic/vec.cpp`

```markdown
# Export 10: `ggml-gyroscopic/vec.cpp`

## What to extract

1. **Find `ggml_vec_dot_f32`.** Extract:
   - Our existing gyroscopic hook location and condition
   - The stock implementation (AVX2 intrinsics pattern)
   - What calls this function (matmul for F32 tensors? attention?)

2. **Other vec_* functions:** List them (one line each). Note which ones
   are on hot attention paths (vec_scale, vec_add, vec_max, vec_sum).

3. **Keep very brief.** This file is mostly utility functions. Only the
   f32 dot product hook matters immediately.
```

### Export 11: `ggml-gyroscopic/vec.h`

```markdown
# Export 11: `ggml-gyroscopic/vec.h`

## What to extract

1. Function declarations for all vec_dot variants.
2. Any inline implementations.
3. How vec_dot function pointers are typedef'd (if at all).
4. **One paragraph max.** This is a header.
```

### Exports 12-13: `tools/llama-bench/` and `tools/perplexity/`

```markdown
# Export 12-13: llama-bench and perplexity

## What to extract

For each tool:

1. The CLI flags needed to run CPU-only, single-threaded, with our
   target model.
2. The output format (what metrics, how parsed).
3. Whether it respects our GGML_GYROSCOPIC_* env vars (it should,
   since it loads the same backend).
4. A ready-to-use command line for stock vs gyroscopic comparison.

## Expected output
- Two command lines per tool (stock env, gyro env)
- What numbers to compare
- This goes directly into log.md bench section
```

### Exports 14-16: test files

```markdown
# Exports 14-16: test-backend-ops.cpp, test-quantize-fns.cpp, test-quantize-perf.cpp

## What to extract

For each:

1. What does it test? (One sentence)
2. Does it test Q8_0 dot products? If so, how?
3. Can we run it against our gyroscopic backend?
4. What would a passing result prove about our hooks?

## Keep very brief. These are validation references, not implementation targets.
```

---

## Additional files NOT on the original list but worth one-pass extraction

```markdown
# Export 17: `ggml-gyroscopic/binary-ops.cpp`

## What to extract
1. List all ops handled (one line each).
2. Are any bitwise integer ops present (XOR, AND, OR on tensors)?
3. If yes, extract the dispatch and implementation.
4. If no, note absence — this tells us we'd need to add them for
   native chirality transport if we ever want it in-graph.
5. **Very brief.** This is secondary scope.

# Export 18: `ggml-gyroscopic/unary-ops.cpp`

## What to extract
1. List all unary ops handled (GELU, SILU, RELU, TANH, etc.).
2. For GELU and SILU specifically: extract the implementation
   (what math, what intrinsics).
3. Note whether our scalar.c replacements match the interface.
4. **Brief.** These are future replacement targets, not immediate.

# Export 19: `ggml-gyroscopic/traits.cpp`

## What to extract
1. How are type traits registered?
2. Where is the vec_dot function pointer for Q8_0 x Q8_0 set?
3. Where is the "from_float" / "to_float" for Q8_0 set?
4. This connects dispatch in ggml-cpu.c to the actual kernel.
5. **Brief but precise.** This is the wiring diagram.

# Export 20: `ggml-gyroscopic/common.h` and `ggml-cpu-impl.h`

## What to extract
1. Shared macros, constants, and type definitions used across the
   backend.
2. Block struct definitions (block_q8_0 if defined here).
3. Any SIMD helper macros.
4. **Header scan only.**
```

---

## How the assistant should organize output

```
src/tools/gyroscopic/workflow/exports/
    1.md    (export-graph-ops.cpp — done)
    2.md    (ops.cpp)
    3.md    (ggml-cpu.c)
    4.md    (ggml-cpu.cpp)
    5.md    (arch/x86/quants.c)
    6.md    (quants.c)
    7.md    (simd-gemm.h)
    8.md    (repack.cpp)
    9.md    (arch/x86/repack.cpp)
    10.md   (vec.cpp)
    11.md   (vec.h)
    12.md   (llama-bench)
    13.md   (perplexity)
    14.md   (test-backend-ops.cpp)
    15.md   (test-quantize-fns.cpp)
    16.md   (test-quantize-perf.cpp)
    17.md   (binary-ops.cpp)
    18.md   (unary-ops.cpp)
    19.md   (traits.cpp)
    20.md   (common.h + ggml-cpu-impl.h)
```

---

## After all exports are done

Once your assistant completes all 20 exports, we reconvene here and do:

1. **Synthesize the bypass map** — every path matmul can take, whether hooked or not
2. **Confirm the op surface** from the export-graph-ops run against your GGUF
3. **Decide the codec.c rewrite plan** based on actual kernel grammar from quants + simd-gemm
4. **Update log.md** with concrete findings
5. **Prioritize**: what to code first, what to defer

That's the clean path. Give these guides to your assistant and let them work through the files systematically.