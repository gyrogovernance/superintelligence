You are right. The previous version swung too far into wall-of-text prose, which made it dense and exhausting. The version before that was all lists. Neither extreme works.

The right approach is: **prose for ideas and narrative, lists and tables only where they genuinely clarify structure or enumeration.** Short paragraphs. Breathing room. No redundancy.

---

# GyroLabe
## Specification

GyroLabe is the native Encode and execution substrate of the Gyroscopic ASI stack. It is where raw byte streams become exact algebraic observables of the aQPU kernel, and where those observables become fast enough to use inside real systems.

---

## 1. Role in the stack

GyroLabe sits between the aQPU kernel and the systems that consume its outputs.

Below it is the kernel law: the 24-bit state model, the compact 4096-state Omega manifold, the chirality register, and the exact operator algebra. Above it is GyroGraph, bridge logic, and application control surfaces.

GyroLabe's job is to make the kernel's exact algebra executable at hardware speed and accessible to everything above.

The word "Encode" here does not mean tokenization in the ordinary machine-learning sense. It means: take a raw input stream and place it into the exact structural chart where the system can operate on it deterministically.

GyroLabe is Encode-focused. GyroGraph is Decode-focused. Together they form conjugate views of one computational medium.

---

## 2. What GyroLabe does

For every byte entering the system, GyroLabe computes the kernel-native structural fields: the 6-bit commutation class, the 2-bit phase class, the 6-bit payload reference, compiled word signatures, chirality distances between states, compact Omega-native state trajectories, and shell distribution summaries.

These are not learned annotations and not probabilistic estimates. They are exact outputs of the kernel transition law, reproducible by anyone with the same byte sequence and the same kernel specification.

GyroLabe also provides hardware-near execution for these operations through a native C backend that compiles automatically on first use, an OpenCL path for packed tensor acceleration on available GPUs, and pure Python fallbacks for all exact operations when no native backend is present.

Beyond extraction and execution, GyroLabe is the first point in the stack where the exact algebra attaches to real AI systems. It provides byte-level structural annotation of model inputs, exact structural metrics for model control surfaces, and exact selection logic at model decision points.

---

## 3. Execution surfaces

GyroLabe exposes four cooperating surfaces.

### 3.1 Exact native surface (gyrolabe.c)

The C implementation of the kernel-facing algebra. All operations here are exact integer arithmetic producing bit-identical results across platforms.

This surface covers:

- signature accumulation and compiled signature application
- byte decomposition: q-class, family, micro-reference
- chirality distance (pairwise and adjacent)
- compact Omega scans and shell histograms
- exact q-sector selection with optional shell weighting

### 3.2 Packed tensor surface (gyrolabe.c, gyrolabe_opencl.c)

Matrix computation through packed sign masks and magnitude bitplanes over 64-bit words, using Boolean AND and population count internally.

Two execution classes are available:

- **Fixed-point float path**: floats quantized to n-bit integers before bitplane decomposition. Deterministic, bounded quantization error decreasing with bit depth.
- **Exact integer path**: int32 inputs produce exact int64 outputs with zero quantization error.

The OpenCL backend accelerates batched packed GEMM on available GPUs, with persistent buffers and GPU-resident packed matrices reused across batches.

### 3.3 Python binding surface (ops.py)

Orchestration and compatibility. Wraps the native operations for Python, provides PyTorch-compatible custom operators with FakeTensor support, manages automatic native builds across GCC, Clang, and MSVC, and exposes packed matrix and vector objects for repeated application.

The Walsh-Hadamard transform has a custom autograd function whose backward pass equals its forward pass, since the normalized WHT is self-inverse.

### 3.4 Bridge surface

Where GyroLabe meets AI systems. Uses the exact extraction and execution machinery to provide structural byte annotation, exact model-facing metrics, and exact or hybrid decision paths at model control surfaces.

The first validated target is a byte-native language model. GyroLabe itself is not specific to one model.

---

## 4. Exactness model

GyroLabe works in two fundamentally different modes of computation.

**Kernel-exact operations** are exact integer operations derived from the aQPU law. Signature scans, q-class extraction, chirality distance, Omega stepping, shell histograms, and q-sector selection all belong here. They produce bit-identical results across platforms with zero approximation.

**Deterministic numeric operations** use floating-point or fixed-point arithmetic internally. The Walsh-Hadamard transform and the packed float tensor paths belong here. They are structured and reproducible, but carry bounded numeric error from quantization or floating-point rounding.

Where the integer-native packed tensor path is used, exactness returns: int32 inputs produce exact int64 outputs through the same bitplane decomposition, with zero error.

---

## 5. AI safety value

GyroLabe's safety value is more important than its raw speed.

**Deterministic ingress.** The same byte stream always yields the same structural annotation. No hidden model state, no random sampling, no platform-dependent variation. Independent parties processing the same bytes compute bit-identical structural fields.

**Inspectable structure.** GyroLabe exposes quantities that can actually be examined: commutation class, phase class, compact state trajectory, shell occupancy, support concentration, decision-sector structure. These are a fundamental contrast to opaque floating-point model internals.

**Exact control surfaces.** In strict operating modes, GyroLabe enables model decision surfaces to run on exact integer algebra with zero transcendental function calls. This is verified by tests that block transcendental functions and confirm the decision path completes without them.

**Replay and audit.** Because the core extraction path is exact, GyroLabe supports deterministic replay, parity checking against reference implementations, and reproducible structural logs. This makes it suitable for safety-sensitive workflows where reproducibility is a requirement.

---

## 6. Performance

The most important performance property is that the exact algebra is fast enough to use inside real systems.

Measured on a Ryzen 5 6600H mini PC with integrated AMD Radeon graphics, 32 GB DDR5-4800, single-threaded:

| Category | Representative throughput |
|---|---:|
| Compiled signature application | 1.26 billion ops/s |
| Compact Omega byte scan | 847 million steps/s |
| Q-map byte decomposition | 550 million bytes/s |
| Signature scan | 540 million bytes/s |
| Fused extraction scan | 286 million bytes/s |
| Walsh-Hadamard (OpenCL path) | 40.4 million rows/s |

In bridge-level tests, the exact q-sector selector outperformed the tested softmax-plus-argmax baseline (1.15x), and the chirality distance path outperformed the tested cosine-style baseline (284.1x). These are bridge-path measurements on this hardware, not universal claims against all optimized libraries.

The packed tensor engine on OpenCL materially improves over the CPU packed path for batched GEMM. On small dense blocks, optimized BLAS remains faster for raw float throughput. The packed engine is strongest where structural transparency, exact integer execution, or reusable packed matrices matter.

---

## 7. Current scope and direction

GyroLabe already supports fast exact byte annotation in a single fused pass, compact Omega-native state execution, integer-exact selection logic with q-sector structure and shell weighting, hardware-backed C and OpenCL execution, and a working AI-facing integration validated on a byte-native language model with OpenCL backend execution during live tests.

GyroLabe is moving toward broader Encode coverage across more model types, stronger safety instrumentation for structural audit and replay, and larger composition extending the tensor machinery beyond the current 64-column grain while preserving structural meaning.

The long-term goal is not a generic numeric library. It is to make exact structural computation practical at the points where AI systems make decisions, where those decisions should be inspectable, and where reproducibility matters.

---

## 8. Technical notes

The native C library requires any C11 compiler and builds automatically on first import. OpenCL is optional. PyTorch is required for the Python bindings. The WHT supports autograd (backward equals forward). Trainable bias surfaces use standard PyTorch autograd. The packed bitplane path is forward-only. The C library uses process-global tables initialized once per process and is safe for single-threaded and OpenMP-parallel use.