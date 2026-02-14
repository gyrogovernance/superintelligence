# GyroLabe: Holographic Coordination Logistics System  
## Technical Specification

### 1. System Overview

GyroLabe couples a generative language model to a finite, discrete geometric reference frame.

The reference frame is the **GGG ASI Alignment Router Kernel**, a deterministic finite-state machine with:

- **States:** 65,536 (2¹⁶)  
- **Actions:** 256 (one for each byte 0..255)  

The kernel does not interpret meaning. It transforms bytes through fixed-width bit operations and exposes geometric observables. The model remains stochastic. GyroLabe biases the model's inference using kernel geometry, while the kernel advances deterministically from the model's sampled tokens.

#### 1.1 Closed loop (per generated token)

At each inference step:

1. **Read kernel observables:** horizon h, vertex charge χ, phase p (and last-byte weight w).  
2. **Projection:** compute a geometry-derived mask and apply it inside routed MLP layers.  
3. **Sampling bias:** re-rank top-k logits using kernel geometry and an intrinsic aperture scale.  
4. **Sample token:** model samples a token (stochastic inference preserved).  
5. **Token → byte → kernel:** driving byte = (token_id & 0xFF) advances the kernel.

#### 1.2 Common Language and byte streams

**Common Language (token → byte)**  
- Driving byte: `byte = token_id & 0xFF`  
- The mapping is **many-to-one**: many tokens share the same driving byte.  
- Every token maps to a valid kernel action.

**Two recorded byte streams**  
- **Driving byte stream:** derived from sampled tokens, advances the kernel.  
- **Extracted byte stream:** derived from activations, recorded as telemetry only. It does not advance the kernel.

#### 1.3 Scope

GyroLabe uses only the kernel physics layer of the router system. It does not include application governance components such as domain ledgers, governance events, coordinator plugins, or external Hodge decompositions.

#### 1.4 Model compatibility (reference implementation)

The GyroLabe formalism is model-agnostic, but the reference implementation targets transformer families with LLaMA/OLMo/Mistral-like structure.

The reference implementation requires:

- **Layer access:** `model.model.layers` exists and is indexable  
- **MLP structure:** routed layers expose a SwiGLU-style MLP with:
  - `gate_proj`, `up_proj`, `down_proj`
- **Divisibility constraints:**
  - `model.config.hidden_size % 256 == 0`
  - the routed MLP hidden dimension (the SwiGLU intermediate) is divisible by 256

If these constraints are not met, the reference implementation raises errors rather than silently degrading.

---

### 2. The Reference Instrument (Kernel)

#### 2.1 State space

The kernel operates on a finite state space Ω with exactly 65,536 states.

Each state is represented as two 12-bit components:

- **A:** active component (12 bits)  
- **B:** passive component (12 bits)

Each 12-bit component corresponds to a 2 × 3 × 2 binary grid.

Let M be the kernel's 256-element 12-bit mask set. Then:

- A_set = { archetype_A ⊕ m : m ∈ M }  
- B_set = { archetype_B ⊕ m : m ∈ M }  
- Ω = A_set × B_set  
- ∣Ω∣ = 256 × 256 = 65,536  

The kernel is **byte-complete**: all bytes 0..255 are valid actions.

#### 2.2 Holographic boundary (Horizon)

The kernel defines a distinguished subset called the Horizon H:

- A state lies on the Horizon when: `A = B ⊕ 0xFFF`  
- There are exactly 256 Horizon states

A verified discrete bijection (the Holographic Dictionary) relates bulk states to boundary structure in the form:

- (horizon anchor, byte) ↔ bulk state  

This is "holographic" in the strict discrete sense of a bijection, not a physical claim.

#### 2.3 Intrinsic invariants

**Intrinsic aperture (A_kernel)**  
The kernel's intrinsic aperture is:

- A_kernel = 5/256 ≈ 0.01953

This value arises from the mask code's weight distribution (probability of masks with weight ≤ 1). In GyroLabe, A_kernel is used as the global scale factor for logit re-ranking, ensuring the kernel's influence remains a small, geometry-grounded perturbation.

**Monodromy defect angle**  
From the minimal nonzero mask weight sector under θ = arccos(1 − w/6):

- δ_kernel ≈ 0.1952 rad

#### 2.4 Trajectory invariants (parity commitment)

For a byte sequence producing masks m₁, m₂, …, mₙ:

- O = m₁ ⊕ m₃ ⊕ m₅ ⊕ …  
- E = m₂ ⊕ m₄ ⊕ m₆ ⊕ …  

The final state depends only on the initial state, O, E, and n mod 2. The ordering of bytes within each parity class is irrelevant. The triple (O, E, parity) provides a compact integrity check for long ledgers.

#### 2.5 Determinism and shared moments

Given the same archetype state and the same driving byte ledger prefix, all conforming implementations compute the same kernel state at each step. This enables "shared moments" defined by the ledger, not by clocks or hidden state.

#### 2.6 Atlas artifacts

The kernel's runtime stepping and observables are provided by atlas artifacts.

**Files (reference atlas build sizes)**  
- `ontology.npy` (262,272 bytes): 65,536 reachable states  
- `epistemology.npy` (67,108,992 bytes): 65,536 × 256 next-state index table  
- `phenomenology.npz` (67,347,289 bytes): observables and lookup tables  

**phenomenology.npz contents**  
The reference implementation expects per-state and per-byte lookup arrays sufficient to provide:

- per-state observables: horizon, vertex, phase  
- per-byte properties: byte_weight, byte_charge  
- coupling table(s): gamma_table  
- optional helper arrays may be present

Kernel stepping at runtime is an O(1) table lookup:

- `state_index = epistemology[state_index, byte]`

**Concurrency**  
The atlas artifacts are immutable after loading. In multi-threaded or multi-process deployments, a single copy of the atlas can be shared via memory mapping (mmap) to minimize RAM usage. This is especially relevant when running multiple model replicas on the same host.

#### 2.7 Verification

The reference implementation is accompanied by unit and integration tests.

- Kernel and atlas correctness are validated by router test suites and exhaustive transition checks during atlas build.  
- GyroLabe coupling logic is validated by tests under `tests/gyrolabe/`, including mask computation, re-ranking stability, and telemetry range checks.  
- Tests requiring atlas artifacts are skipped when artifacts are not present.

---

### 3. The Coordination Cycle

#### 3.1 Prompt priming

The kernel may be primed by stepping through prompt token IDs before generation. In that case, the driving byte ledger includes both prompt bytes and generated-token bytes.

Reference implementation: `prime_from_tokens()` in `src/tools/gyrolabe.py`.

#### 3.2 Kernel observables

At each step t, GyroLabe reads these kernel observables:

- **Horizon index hₜ ∈ [0, 255]**  
  Primary coordinate for projection and re-ranking.

- **Vertex charge χₜ ∈ [0, 3]**  
  K₄ vertex class for the current position; used as a wedge selector and width selector.

- **Phase pₜ ∈ [0, 3]**  
  Cycle phase of the last action; used as a modulation factor (not as a mask shift) in the reference implementation.

- **Last-byte weight wₜ ∈ [0, 12]**  
  Hamming weight of the 12-bit mask of the most recently applied driving byte; scales projection strength.

#### 3.3 Projection (kernel → model activations)

The reference implementation applies a mask inside routed MLP layers to the SwiGLU hidden activation:

- z = silu(gate_proj(x)) ⊙ up_proj(x)

The intermediate hidden dimension is factored as:

- hidden_dim = 256 × N_feat  
  (example: OLMo uses hidden_dim = 11008 = 256 × 43)

**Boundary index interpretation**  
Boundary index x ∈ {0..255} is treated as a byte-labeled code element. Distances are computed as Hamming distance in the kernel's 12-bit mask code.

**Mask computation (reference implementation behavior)**  
- The mask is centered in code space at the current horizon index hₜ via distances = code_dist_matrix[hₜ, x].  
- Phase pₜ modulates mask width (sigma), not the mask center.

A readable step summary:

1. **Code-space radius:** compute d(x) = HammingDistance(mask(hₜ), mask(x))  
2. **Gaussian base:** mask_base(x) = exp(−0.5 · (d(x)/σ)²), with σ selected by χₜ and modulated by pₜ  
3. **Vertex wedge:** boost positions whose byte charge matches χₜ  
4. **Strength scaling:** blend toward 1 using a factor derived from wₜ  
5. **Normalization:** scale so mean(mask) = 1 and broadcast across N_feat

Reference implementation: `compute_mask()` and `RoutedMLP` in `src/tools/gyrolabe.py`.

#### 3.4 Logit re-ranking (kernel → sampling surface)

Before sampling, GyroLabe may re-rank the model's top-k logits using kernel geometry.

**Scope and interface**  
- Re-ranking is applied **only to top-k candidates**.
- Candidate tokens are mapped to bytes by `byte = token_id & 0xFF`.

**Scoring components**  
- **Code distance:** d = HammingDistance(mask(hₜ), mask(byte))
- **Alignment score:** align = 1.0 − (d / 12.0)
- **Vertex compatibility:** bonus if candidate's byte charge matches χₜ
- **Gamma interaction:** Γ[χ, q, w] is a precomputed tensor stored in the atlas. It encodes structural compatibility between the current vertex class χ, the candidate's byte charge q, and the candidate's mask weight w.

**Adjustment scale**  
- Δlogit is proportional to A_kernel and the spread of the top-k logits.
- This keeps the effect a small perturbation, not an override.

Reference implementation: `_rerank_topk_logits_kernel_native()` in `src/tools/gyrolabe.py`.

#### 3.5 Token advancement (model → kernel)

After sampling:

- driving_byteₜ = token_idₜ & 0xFF  
- `kernel.step_byte(driving_byteₜ)`

This is the only mechanism by which the kernel advances in the reference implementation.

Reference implementation: `advance_with_token()` in `src/tools/gyrolabe.py`.

#### 3.6 Extraction telemetry (model → analysis only)

GyroLabe extracts telemetry from activations to measure coupling quality. This telemetry does not affect kernel stepping.

**Extraction mechanism**  
The extraction of a byte from the activation peak follows a specific geometric heuristic:

1. Reshape output activation to boundary form [batch, 256, N_fiber].
2. Compute energy (squared magnitude) at each of the 256 boundary positions.
3. Identify the peak boundary index h_peak with maximal energy.
4. Select the fiber vector at that position (dimension N_fiber).
5. Form the extracted byte from the **sign pattern** of fiber channels: bit i is 1 if channel[2i] > 0, else 0.

**Telemetry metrics**  
- **h_peak:** boundary index with maximum energy
- **peak_mass:** concentration at the peak
- **correlation:** cosine similarity between energy distribution and the applied mask  
- **code_dist:** Hamming distance in mask code space between hₜ and h_peak  

**Diagnostic coordinate μ (telemetry only)**  
The telemetry also computes:

- μₜ = (hₜ + 64 · pₜ) mod 256  
- dist_to_μ = circular distance from h_peak to μₜ  
- This μₜ is a diagnostic phase-lifted coordinate. It is not the center of the projection mask in the reference implementation.

Reference implementation: `extract_byte()`, `begin_step()`, `end_step()` in `src/tools/gyrolabe.py`.

#### 3.7 Code distance baseline (why "near 6" is neutral)

The kernel mask code C is a linear [12, 8] code with a palindromic weight distribution centered at weight 6. For two masks drawn without structural bias from C, the expected Hamming distance is 6 bits.

Interpretation:

- **code_dist ≈ 6 over long trajectories** indicates the interaction preserves the intrinsic isotropy of the code space.  
- **Systematic deviation away from 6** (as an average over many steps) indicates directional bias in how the coupled system explores code space.

---

### 4. Reference Implementation (files and responsibilities)

This specification corresponds to the reference implementation organized as follows:

- **Kernel stepping and observables:** `src/router/kernel.py` (uses atlas artifacts)  
- **Kernel-derived constants and helpers:** `src/router/constants.py`  
- **Coupling system and generation loop:** `src/tools/gyrolabe.py`  
  - `GyroLabe` (wrapping, telemetry, stepping)
  - `RoutedMLP` (projection injection)
  - `compute_mask` (projection mask)
  - `_rerank_topk_logits_kernel_native` (top-k re-ranking)
  - `generate` (sampling loop with optional coupling)
- **Runner with diagnostics:** `scripts/run_gyrolabe.py`

Configuration surface:

- `CouplingConfig.routed_layers`: explicit layer list (or auto-detect)  
- `CouplingConfig.store_layer_telemetry`: enable or suppress per-layer telemetry storage  

---

### 5. Metrics and interpretation

GyroLabe exposes summary statistics over a trajectory.

#### 5.1 Kernel and ledger metrics

- steps processed  
- kernel signature (state hex and components)  
- parity invariants: O, E, parity  

#### 5.2 Distribution metrics

- horizon coverage: unique_h, horizon entropy  
- vertex distribution: χ histogram  
- driving byte distribution: unique bytes, byte entropy  
- mask weight histogram (from driving bytes)

#### 5.3 Alignment metrics

- mean_code_dist (baseline near 6.0 is neutral isotropy)  
- mean_correlation (energy distribution vs mask)  
- mean_gain_at_peak (mask value at peak, mask is mean-normalized to 1)  
- distances to diagnostic μ (phase-lifted coordinate, telemetry only)

#### 5.4 Quality metrics (logprobs and perplexity)

In the reference implementation:

- logprobs are computed from the model's **unmodified logits** (before re-ranking)
- sampling may occur from a **re-ranked** top-k distribution

Interpretation:

- These logprobs measure "how probable were the sampled tokens under the base model distribution," not likelihood under the re-ranked distribution.

---

### 6. Topological alignment

**Definition**  
Topological alignment is the condition where the coupled model–kernel dynamics preserve the intrinsic metric structure of the kernel mask code C, where the intrinsic metric is **Hamming distance on the 12-bit codewords**.

Operational evidence (observed together, over long runs):

- code_dist remains near the code's natural 6-bit baseline  
- correlation is consistently positive beyond null baselines  
- horizon and byte entropies remain high (exploration is not collapsed)

---

### 7. Application scenarios

#### 7.1 Generative guidance

The kernel biases generation through:

- **internal coupling:** projection mask inside routed MLP layers  
- **output coupling:** geometry-aware top-k re-ranking

The model remains stochastic. The kernel provides a stable finite reference geometry.

#### 7.2 Distributed synchronization

Participants share a byte ledger. Replaying the kernel from the archetype through the shared ledger yields identical states, enabling coordination without trusted timestamps.

#### 7.3 Provenance and audit

A byte-encoded operation history yields a reproducible kernel trajectory and compact parity invariants. Insertions, deletions, or reorderings change the invariants.

---

### 8. Analytical note

The kernel's 256-element boundary can be lifted to a 256-dimensional vector space representation. In this view, GyroLabe can be analyzed as a structural resonance between continuous model dynamics and a discrete, deterministic code geometry. Telemetry provides empirical measures of coupling strength (correlation) and geometric neutrality (code_dist baseline behavior).

---

### 9. Operational considerations

#### 9.1 Batch vectorization

Production inference operates on batches of sequences (batch size B > 1). GyroLabe supports full vectorization:

- **Kernel state:** The kernel maintains a state vector of size B. Each sequence in the batch evolves an independent trajectory.
- **Atlas lookups:** Next-state transitions and observable lookups are vectorized integer gather operations.
- **Mask computation:** The projection mask becomes a tensor of shape [B, 256, 1], computed in parallel for the (hₜ, χₜ, pₜ, wₜ) tuple of each sequence.
- **Telemetry:** Per-sequence extraction and metric collection are independent and parallelizable.

#### 9.2 Computational overhead

The coordination cycle adds minimal latency to the inference loop:

- **Kernel stepping:** O(1) integer array lookup per sequence. Negligible compared to matrix multiplications in transformer layers.
- **Mask computation:** Gaussian over 256 positions, then element-wise multiplication on the hidden state. Cost is proportional to activation size, typically < 1% of layer compute.
- **Re-ranking:** Operations are performed only on the top-k candidates (e.g., k = 40), not the full vocabulary. Cost is negligible relative to the forward pass.

The dominant cost remains the model's own forward and backward passes. GyroLabe overhead is not measurable at typical inference scales.

#### 9.3 Precision and determinism

**Kernel (integer domain)**  
- Uses uint16/uint32 arithmetic throughout.
- Bit-exact across all hardware platforms (CPU, CUDA, MPS).
- Ledger replay produces identical states on any conforming implementation.

**Projection and re-ranking (floating-point domain)**  
- Uses standard floating-point math (exp, division, multiplication).
- Minor numerical divergence across platforms is expected due to FP associativity and fused operations.
- This divergence does **not** affect the discrete kernel state, which depends only on the driving byte (derived from the sampled token_id, an integer).

#### 9.4 Memory considerations

**Atlas sharing**  
- The atlas is read-only after loading.
- In multi-process deployments (e.g., multiple model replicas on one host), the atlas can be memory-mapped (mmap) so all processes share a single physical copy.
- Total atlas footprint: approximately 128 MiB.

**Per-sequence state**  
- Each active sequence requires storage for one kernel state index (uint32) plus any accumulated telemetry.
- For B concurrent sequences without stored telemetry: 4B bytes of kernel state.

#### 9.5 Failure modes

- **Missing atlas:** The kernel raises an error at construction if atlas files are not found. Generation cannot proceed.
- **Incompatible model architecture:** GyroLabe raises an error during `install()` if the model does not expose the required interface or fails divisibility checks.
- **Numeric instability:** If mask computation produces NaN or Inf (e.g., due to extreme sigma values), the reference implementation clamps or skips the mask for that step. This is logged but does not halt generation.