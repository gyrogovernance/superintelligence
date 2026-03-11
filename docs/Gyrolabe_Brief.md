

# GyroLabe Brief

## What GyroLabe Is

GyroLabe is the native compute backend and model bridge for the Gyroscopic ASI aQPU Kernel. It is a C/OpenCL library with Python bindings that provides two capabilities:

1. A fast algebraic execution layer for the aQPU kernel (signature scanning, chirality distance, q-map extraction, Walsh-Hadamard transforms, state stepping).

2. A bitplane matrix-vector multiplication engine that decomposes dense floating-point linear algebra into Boolean AND + POPCNT operations over 64-bit words, with a bridge that connects this engine to real language model weights.

The first capability accelerates kernel-native operations by 1,000x to 8,000x over Python. The second capability is the beginning of a path toward structurally transparent neural network inference.

GyroLabe is not finished. This brief describes what exists, what it does, what it does not yet do, and what the development trajectory looks like.

---

## Architecture

GyroLabe has four layers:

**Layer 1: C ALU (gyrolabe.c)**

Plain C, cross-platform, ctypes-friendly. Exports exact integer kernel operations:

- `gyro_signature_scan`: accumulates word signatures over byte ledgers
- `gyro_chirality_distance`: Hamming distance on collapsed 6-bit chirality words
- `gyro_qmap_extract`: per-byte q-class, family, and micro-reference extraction
- `gyro_extract_scan`: fused single-pass extraction of all the above plus states
- `gyro_wht64_float`: 64-point Walsh-Hadamard transform (orthonormal, 1/8 scaling)
- `gyro_step_byte_batch`, `gyro_state_scan_from_state`: batched state stepping
- `gyro_apply_signature_to_rest`, `gyro_apply_signature_to_state`, `gyro_apply_signature_batch`: compiled word application

All kernel math is exact integer. The WHT uses float arithmetic with the standard butterfly decomposition.

The same file contains the bitplane GEMV engine:

- `gyro_bitplane_gemv_f32`: full float-to-fixed-to-bitplane GEMV in one call
- `gyro_pack_bitplane_matrix_f32` / `gyro_pack_bitplane_vector_f32`: pack-once, multiply-many path
- `gyro_bitplane_gemv_packed_f32`, `gyro_bitplane_gemv_packed_x_f32`: packed GEMV variants
- `gyro_bitplane_gemm_packed_x_batch_f32`: batched GEMM with OpenMP support
- Integer-native variants (`_i32`): no quantization, no scaling, exact int32-to-int64 multiplication

The bitplane engine works on matrices with up to 64 columns. It quantizes floats to fixed-point integers, decomposes them into sign masks and magnitude bitplanes (one uint64 per bitplane per row), and computes dot products using AND + POPCNT across bitplane pairs. The inner loop is O(n_bits^2) POPCNT operations per row, where n_bits is the quantization depth (8, 12, or 16).

**Layer 2: OpenCL Backend (gyrolabe_opencl.c)**

GPU-accelerated batched GEMM. The OpenCL backend handles only the dense multiply. Everything else (signatures, q-map, chirality, control plane) stays on CPU.

The OpenCL kernels mirror the CPU bitplane logic: each work-item computes one (row, batch) element using the same AND + POPCNT decomposition. Specialized 64x64 kernels use local memory to cache the input vector bitplanes per workgroup. Kernels are compiled at init time for n_bits in {8, 12, 16}, for both float32 and int32 paths.

Persistent transient buffers are reused across GEMM calls to avoid repeated allocation. GPU-resident packed matrices are uploaded once and reused across many input batches.

**Layer 3: Python Ops (ops.py)**

PyTorch-integrated Python layer. Provides:

- torch.library custom ops (`gyro::signature_scan`, `gyro::chirality_distance`, `gyro::wht64_forward`, `gyro::qmap_extract`) with FakeTensor support for torch.compile
- Pure-Python fallbacks for all kernel-exact operations when the C library is unavailable
- `PackedBitplaneMatrix64` and `PackedBitplaneVector64` classes (float path)
- `PackedBitplaneMatrix64I32` and `PackedBitplaneVector64I32` classes (integer-native path)
- `OpenCLPackedMatrix64` and `OpenCLPackedMatrix64I32` for GPU-resident matrices
- Automatic C library build, compiler detection (GCC, Clang, MSVC), and DLL path management

The WHT has a custom autograd function: backward = forward, since the normalized WHT is self-inverse and orthonormal.

**Layer 4: Bolmo Bridge (bolmo_bridge.py)**

The connection to a real language model. Currently targets Bolmo-1B, a byte-level causal language model. The bridge:

- Wraps a HuggingFace Bolmo model as a PyTorch module
- Installs hooks on the local encoder's byte embedding layer and boundary predictor
- Canonicalizes Bolmo token IDs to raw bytes [0..255] accounting for the model's offset and fused-token structure
- Runs `extract_scan` on the canonical byte sequence to get q-class, family, micro-reference, signatures, and states for every position
- Adds trainable embedding biases (q_class_embedding, family_embedding, micro_ref_embedding) to the byte embedding output, gated by a validity mask
- Adds a trainable boundary distance bias derived from chirality distance between adjacent kernel states
- Optionally provides q-class sparsity masks for attention (currently disabled by default)

The bridge is designed to be lossless at initialization: all GyroLabe parameters start at zero, so the wrapped model produces identical outputs to the unwrapped model before any training.

---

## Current Performance

From the benchmark suite (CPU: standard desktop, GPU: consumer OpenCL device):

**Kernel operations (C vs Python):**

| Operation | n | Python | C | Speedup |
|-----------|---|--------|---|---------|
| signature_scan | 65,536 | 420 ms | 0.12 ms | 3,376x |
| chirality_distance | 262,144 | 2,549 ms | 1.63 ms | 1,568x |
| qmap_extract | 65,536 | 1,010 ms | 0.12 ms | 8,222x |

**Chirality distance vs cosine similarity (2048-dim):**

| n | chirality | cosine | speedup |
|---|-----------|--------|---------|
| 16,384 | 0.12 ms | 41.7 ms | 353x |

The chirality distance comparison is relevant because it shows the structural distance metric operating 2-3 orders of magnitude faster than the standard similarity metric used in transformer attention, while carrying algebraically characterized information (exact chirality transport, q-class membership).

**Bitplane GEMV/GEMM (64x64 blocks):**

| Path | Error vs torch.mv |
|------|-------------------|
| Random matrix, C unpacked | ~1.1e-05 |
| Real Bolmo q_proj block, C unpacked | ~8.4e-07 |
| Random matrix, C packed | ~7.5e-06 |
| Real Bolmo q_proj block, C packed | ~3.6e-07 |
| Identity matrix, C packed | ~3.5e-05 |

**Batched GEMM (64x64, batch scaling):**

| Batch | torch | CPU packed | OpenCL | OpenCL vs torch |
|-------|-------|------------|--------|-----------------|
| 256 | 1.34 ms | 3.89 ms | 1.00 ms | 1.3x faster |
| 1,024 | 5.99 ms | 13.27 ms | 3.74 ms | 1.6x faster |
| 4,096 | 21.95 ms | 53.57 ms | 14.29 ms | 1.5x faster |

**Integer-native OpenCL (exact, no quantization error):**

| Batch | CPU vs OpenCL error |
|-------|---------------------|
| 64 | 0 (exact) |

The OpenCL path exceeds PyTorch at batch >= 256 for 64x64 blocks. The integer-native path produces bit-exact results.

---

## What the Bolmo Bridge Does Today

The bridge has been tested on Bolmo-1B (a byte-level transformer). Here is what it currently provides:

**Byte-level algebraic annotation.** Every byte position in the input sequence gets its q-class (6-bit commutation invariant), family (2-bit spinorial phase), micro-reference (6-bit payload), kernel signature, and kernel state. This is computed in a single fused pass through the C backend.

**Trainable algebraic embedding bias.** Three embedding tables (q_class: 64 entries, family: 4 entries, micro_ref: 64 entries) add algebraic structure to the byte embedding. These are initialized to zero (lossless at init) and can be trained to let the model learn to use the aQPU's structural decomposition of bytes.

**Boundary prediction bias.** The chirality distance between adjacent kernel states provides a structural signal for Bolmo's boundary predictor (which decides where to segment byte sequences into tokens). A trainable embedding maps chirality distances [0..6] to scalar biases added to the boundary log-probabilities.

**Q-class attention sparsity (experimental, disabled).** When enabled, the bridge constructs a sparse attention mask where position i attends to position j only if their q-classes match. Since commutativity rate is 1/64, this would produce an extremely sparse mask. This is currently disabled because the interaction with Bolmo's sliding/full attention patterns needs further work.

**Decode expansion cache.** An optional cache for the tokenizer's byte-to-token expansion during autoregressive generation. Output-neutral: identical results, potentially faster decoding.

---

## What GyroLabe Does Not Yet Do

**Dimensional scaling.** The bitplane engine operates on matrices with up to 64 columns. Real transformer layers have dimensions 768, 1024, 2048, 4096, or larger. Running a full transformer layer through the aQPU requires tiling the weight matrices into 64-column blocks. The tiling infrastructure does not exist yet. The core question is whether the algebraic properties (chirality transport, q-class structure, K4 gate decomposition) compose meaningfully across tiles, or whether they fragment.

**Operator decomposition.** The SDK specifies an operator projection basis (Weyl/Heisenberg-Walsh decomposition) that decomposes any 64x64 real matrix into the aQPU's native operator algebra. The test suite verifies project-reconstruct exactness at full rank and graceful degradation at lower rank. But the bridge does not yet decompose Bolmo's weight matrices into this basis. This is the step that would make each inference operation structurally legible in terms of the kernel's algebraic vocabulary.

**Training loop.** The GyroLabe parameters (embedding biases, boundary distance bias) are trainable PyTorch parameters with proper autograd support (the WHT has a correct backward pass). But no training loop, loss function, or fine-tuning recipe exists yet. The hypothesis is that a short fine-tuning pass with the algebraic biases active would let the model learn to exploit the byte structure, but this has not been tested.

**Inference-time algebraic audit.** The architecture is designed so that inference through the aQPU produces a byte ledger, Shared Moments, and governance-measurable trajectories. The plumbing for this exists at the kernel level (signature scan, chirality tracking, parity commitments), but it has not been connected to actual inference runs in a way that produces interpretable audit trails.

**Multi-model support.** The bridge is written for Bolmo-1B specifically. Generalizing to other byte-level models (or to subword models with byte fallback) requires adapter work.

**WHT as attention alternative.** The Walsh-Hadamard transform is the spectral dual of the chirality register and could in principle serve as an attention mechanism (it is exact, invertible, O(n log n), and native to the kernel). This is a research direction, not a tested capability.

---

## The Trajectory

GyroLabe sits at the junction between three things that exist independently:

1. A verified algebraic kernel (4,096 states, self-dual code, K4 gates, exact chirality transport, proven computational advantages) with a complete SDK.

2. A bitplane multiplication engine that can execute dense linear algebra through Boolean operations at competitive speed, with both float-quantized and integer-exact paths.

3. A working bridge to a real byte-level language model that injects the kernel's algebraic decomposition of bytes into the model's processing pipeline.

The development path is:

**Near term:** Tiling infrastructure for the bitplane engine to handle full model dimensions. Training loop for the algebraic embedding biases. Measurement of whether the biases improve perplexity or calibration on standard benchmarks.

**Medium term:** Operator decomposition of weight matrices into the aQPU basis. Per-layer algebraic audit: for each transformer layer, what is the distribution of operator types, q-classes, and gate phases? Does this distribution correlate with known model behaviors (in-context learning, factual recall, reasoning)?

**Longer term:** Inference-time audit trails. Structural interpretability: can the algebraic decomposition identify which operators drive specific model capabilities? Governance-measurable inference: can the aperture and Hodge decomposition from the governance layer be applied to the inference trace itself?

The goal is not to replace neural network inference with algebraic computation. It is to make neural network inference run through an algebraic medium where every operation is decomposable, every step is replayable, and the structural properties of the computation are auditable by anyone with the byte ledger and the kernel specification.

---

## Technical Notes

**Build requirements.** Any C11 compiler (GCC, Clang, MSVC). OpenCL SDK for GPU acceleration (optional). PyTorch for the Python bindings. The C library is built automatically on first import; no manual build step is needed.

**Quantization precision.** The bitplane engine uses n_bits-deep fixed-point quantization. At n_bits=16, typical max errors are ~1e-5 for random matrices and ~1e-6 for real model weights. The integer-native path (i32) has zero quantization error.

**Bolmo compatibility.** The bridge includes a ROPE patch for Bolmo's custom rotary embedding initialization. It loads models locally (no HuggingFace Hub download by default) and handles Bolmo's byte offset/fused-token vocabulary structure.

**Autograd.** The WHT custom op has a correct backward pass (WHT is self-inverse, so backward = forward). The embedding biases are standard PyTorch Embedding modules with standard autograd. The bitplane GEMV does not currently have autograd support; it is forward-only.

**Thread safety.** The C library uses process-global lookup tables initialized once (`gyro_init`). The OpenCL backend uses process-global runtime state. Both are safe for single-threaded use and for OpenMP-parallel batched GEMM. Multi-process use requires separate library instances.