# GyroLabe: An AI Mechanistic Calibration Instrument
**Neural Activation Coordination and Guidance**

## 1. Technical Specification - Overview

GyroLabe is a **mechanistic calibration instrument** for artificial intelligence models. It functions as a **feedback control system**, providing active balance support to stabilize neural activations during inference.

The system acts as a **neuro-symbolic bridge**. It couples the model's stochastic, continuous latent space to a discrete, finite geometric structure.

By imposing a rigid geometric reference frame onto the fluid activations of the model, GyroLabe achieves two objectives:

1.  **Activation Dynamic Stability:** It constrains the model's generation trajectory, preventing collapse or drift (hallucination) by anchoring it to a deterministic state machine.
2.  **Mechanistic Balance Steering:** It injects calculated "weight" into specific neural pathways, ensuring that the model's output remains structurally consistent with the reference geometry.

Unlike training-based alignment (RLHF), GyroLabe operates purely at inference time. It does not alter the model's weights but rather steers the flow of information through the network. It functions analogously to the cerebellum in biological intelligence: it does not decide what to say, it provides a continuous geometric reference frame so that the trajectory of inference remains physically coordinated. The cerebellum was the first computational structure to emerge in vertebrate nervous systems, predating cognition by hundreds of millions of years, and it operates entirely in 3D space with 6 degrees of freedom, which is exactly the geometry the GyroLabe encodes.

**Active Inference**
GyroLabe implements active inference in the precise sense of the term: a system that selects actions by minimizing the divergence between predicted and preferred outcomes under a generative model. The kernel is the generative model (deterministic, finite, exact). The horizon is the Markov blanket (the boundary between the system and its environment). The mask is the descending prediction (what the kernel expects of the model's activations). The token choice is the action (selected by consulting where each candidate would drive the kernel trajectory). The correlation metric is the prediction error signal.

Both halves of active inference are implemented:

- **Perceptual inference (BU-Egress):** the mask biases the model's internal activations toward configurations consistent with the kernel's current geometric state.
- **Active inference (BU-Ingress):** token selection consults the kernel's transport prior to favor actions that produce geometrically natural transitions.

The generative model's prior over actions is not learned or tuned. It is P_code: the exact distribution of pairwise code distances in the kernel's mask code, a structural invariant with no free parameters. The precision of the active inference is beta = A_kernel = 5/256, the kernel's intrinsic aperture. Both the prior and the precision are determined entirely by the kernel's geometry.

**Objective:**
The primary goal is to constrain the high-entropy "hallucination" space of generative models without sacrificing their creative capability. GyroLabe ensures that while the content remains flexible, the **trajectory of activations** adheres to a stable, reproducible geometric logic.

---

### 1.1 Routing

GyroLabe couples a generative language model to a finite, discrete geometric reference frame.

The reference frame is the **GGG ASI Alignment Router Kernel**, a deterministic finite-state machine with:

- **States:** 65,536 (2^16)
- **Actions:** 256 (one for each byte 0..255)

The kernel does not interpret meaning. It transforms bytes through fixed-width bit operations and exposes geometric observables. The model remains stochastic. GyroLabe biases the model's inference using kernel geometry, while the kernel advances deterministically from the model's sampled tokens.

#### 1.2 Closed loop (per generated token)

At each inference step:

1. **Read kernel observables:** horizon h, vertex charge chi, phase p, and last-byte weight w.
2. **Projection (BU-Egress):** compute a geometry-derived mask with CGM-grounded sigma, dynamic sigma-focusing, and differential modulation. Apply it inside routed MLP layers at the SwiGLU intermediate activation.
3. **Active Inference (BU-Ingress):** for each top-k candidate token, peek at the resulting kernel horizon if that byte were applied. Compute a geometric cost from the kernel-native code distance prior P_code. Adjust logits by this cost scaled by the kernel intrinsic aperture beta = 5/256.
4. **Sample token:** model samples a token from the adjusted distribution (stochastic inference preserved).
5. **Token to byte to kernel:** driving byte = (token_id & 0xFF) advances the kernel.

#### 1.3 Common Language and byte streams

**Common Language (token to byte)**
- Driving byte: `byte = token_id & 0xFF`
- The mapping is **many-to-one**: many tokens share the same driving byte.
- Every token maps to a valid kernel action.

**Two recorded byte streams**
- **Driving byte stream:** derived from sampled tokens, advances the kernel.
- **Extracted byte stream:** derived from activations, recorded as telemetry only. It does not advance the kernel.

#### 1.4 Scope

GyroLabe uses only the kernel physics layer of the router system. It does not include application governance components such as domain ledgers, governance events, coordinator plugins, or external Hodge decompositions.

#### 1.5 Model compatibility (reference implementation)

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

The kernel operates on a finite state space Omega with exactly 65,536 states.

Each state is represented as two 12-bit components:

- **A:** active component (12 bits)
- **B:** passive component (12 bits)

Each 12-bit component corresponds to a 2 x 3 x 2 binary grid.

**Topology semantics (normative):**

For every 12-bit component, indices are interpreted as `[frame][row][col]`:

- `row` selects axis family: row 0 = X, row 1 = Y, row 2 = Z.
- `col` selects oriented side of that axis: col 0 = negative side, col 1 = positive side.
- `frame` selects chirality layer: frame 0 and frame 1 are opposing orientation layers of the same 3-axis structure.

Under this convention:

- "3D" refers to the three axis families X, Y, Z.
- "6DoF" refers to six oriented axis sides `{X-, X+, Y-, Y+, Z-, Z+}` defined by row/col pairs.
- Frames do not introduce extra spatial dimensions; they encode opposing chirality used by the kernel transition law.

Let M be the kernel's 256-element 12-bit mask set. Then:

- A_set = { archetype_A XOR m : m in M }
- B_set = { archetype_B XOR m : m in M }
- Omega = A_set x B_set
- |Omega| = 256 x 256 = 65,536

The kernel is **byte-complete**: all bytes 0..255 are valid actions.

#### 2.2 Holographic boundary (Horizon)

The kernel defines a distinguished subset called the Horizon H:

- A state lies on the Horizon when: `A = B XOR 0xFFF`
- There are exactly 256 Horizon states

A verified discrete bijection (the Holographic Dictionary) relates bulk states to boundary structure in the form:

- (horizon anchor, byte) <-> bulk state

This is "holographic" in the strict discrete sense of a bijection, not a physical claim.

#### 2.3 Intrinsic invariants

**Intrinsic aperture (A_kernel)**
The kernel's intrinsic aperture is:

- A_kernel = 5/256 approximately 0.01953

This value arises from the mask code's weight distribution (probability of masks with weight at most 1). A_kernel characterizes the minimal sector of the defect weight distribution and serves as a structural invariant of the kernel physics.

**Monodromy defect angle**
From the minimal nonzero mask weight sector under theta = arccos(1 - w/6):

- delta_kernel approximately 0.1952 rad

#### 2.4 Trajectory invariants (parity commitment)

For a byte sequence producing masks m_1, m_2, ..., m_n:

- O = m_1 XOR m_3 XOR m_5 XOR ...
- E = m_2 XOR m_4 XOR m_6 XOR ...

The final state depends only on the initial state, O, E, and n mod 2. The ordering of bytes within each parity class is irrelevant. The triple (O, E, parity) provides a compact integrity check for long ledgers.

#### 2.5 Determinism and shared moments

Given the same archetype state and the same driving byte ledger prefix, all conforming implementations compute the same kernel state at each step. This enables "shared moments" defined by the ledger, not by clocks or hidden state.

#### 2.6 Atlas artifacts

The kernel's runtime stepping and observables are provided by atlas artifacts.

**Files (reference atlas build sizes)**
- `ontology.npy` (262,272 bytes): 65,536 reachable states
- `epistemology.npy` (67,108,992 bytes): 65,536 x 256 next-state index table
- `phenomenology.npz` (67,347,289 bytes): observables and lookup tables

**phenomenology.npz contents**
The reference implementation expects per-state and per-byte lookup arrays sufficient to provide:

- per-state observables: horizon, vertex, phase
- per-byte properties: byte_weight, byte_charge
- optional helper arrays may be present

Kernel stepping at runtime is an O(1) table lookup:

- `state_index = epistemology[state_index, byte]`

**Concurrency**
The atlas artifacts are immutable after loading. In multi-threaded or multi-process deployments, a single copy of the atlas can be shared via memory mapping (mmap) to minimize RAM usage. This is especially relevant when running multiple model replicas on the same host.

#### 2.7 Verification

The reference implementation is accompanied by unit and integration tests.

- Kernel and atlas correctness are validated by router test suites and exhaustive transition checks during atlas build.
- GyroLabe coupling logic is validated by tests under `tests/gyrolabe/`, including mask computation and telemetry range checks.
- Tests requiring atlas artifacts are skipped when artifacts are not present.

---

### 3. The Coordination Cycle

#### 3.1 Prompt priming

The kernel may be primed by stepping through prompt token IDs before generation. In that case, the driving byte ledger includes both prompt bytes and generated-token bytes.

Reference implementation: `prime_from_tokens()` in `src/tools/gyrolabe.py`.

#### 3.2 Kernel observables

At each step t, GyroLabe reads these kernel observables:

- **Horizon index h_t in [0, 255]**
  Primary coordinate for projection.

- **Vertex charge chi_t in [0, 3]**
  K_4 vertex class for the current position; used as a wedge selector and width selector.

- **Phase p_t in [0, 3]**
  Cycle phase of the last action; used as a modulation factor (not as a mask shift) in the reference implementation.

- **Last-byte weight w_t in [0, 12]**
  Hamming weight of the 12-bit mask of the most recently applied driving byte; scales projection strength.

#### 3.3 Projection (kernel to model activations)

The reference implementation applies a mask inside routed MLP layers to the SwiGLU hidden activation:

- z = silu(gate_proj(x)) * up_proj(x)

The intermediate hidden dimension is factored as:

- hidden_dim = 256 x N_feat
  (example: OLMo uses hidden_dim = 11008 = 256 x N_feat where N_feat is determined by the model architecture)

**Boundary index interpretation**
Boundary index x in {0..255} is treated as a byte-labeled code element. Distances are computed as Hamming distance in the kernel's 12-bit mask code.

**Gaussian lookup table**
The Gaussian base values are precomputed for all (chi, p, distance) triples. There are 4 chi values, 4 phase values, and 13 possible distances, giving 208 precomputed floats. At runtime, the mask is constructed by table lookup rather than computing the exponential function.

**CGM-grounded sigma values**
The base width sigma of the Gaussian mask is derived from the Common Governance Model stage actions relative to the aperture scale m_a = 1 / (2 * sqrt(2 * pi)):

- chi=0 (CS stage): sigma = (pi/2) / m_a / 2, approximately 3.94
- chi=1 (UNA stage): sigma = (1/sqrt(2)) / m_a, approximately 3.55
- chi=2 (ONA stage): sigma = (pi/4) / m_a, approximately 3.94
- chi=3 (BU stage): sigma = 24.0 (near-uniform, balance)

These values are not empirical tuning parameters. They are the dimensionless accumulated phases of the four CGM stage transitions, each normalized by the aperture scale. They determine how tightly the mask couples the model to the kernel geometry at each vertex class.

**Dynamic sigma-focusing**
The effective distance used to sample the Gaussian LUT is scaled by the kernel's motion between consecutive steps:

- D = code_dist(prev_h, h) / 12, normalized to [0, 1]
- sigma_scale = 1.0 - 0.4 * (D - 0.5), clamped to [0.30, 2.50]
- d_eff = d / sigma_scale

When the kernel makes a large jump in code space (high D), sigma_scale decreases, narrowing the effective mask. When the kernel moves a short distance, the mask widens slightly. This ensures that guidance is sharpest precisely when the kernel's geometry is changing most rapidly.

**Mask computation (reference implementation behavior)**
- The mask is computed once per token step in `begin_step()`, then broadcast to all routed MLP layers. This avoids redundant computation.
- The mask is centered in code space at the current horizon index h via distances = code_dist_matrix[h, x].
- Phase p modulates mask width (sigma), not the mask center.
- The Gaussian base is looked up from the precomputed table indexed by (chi, p, distance).
- The byte charge table used for the vertex wedge is the kernel's own `byte_charge` array, loaded from the atlas. GyroLabe does not rebuild this table.

**Differential modulation**
GyroLabe tracks the previous horizon index across steps. When the previous horizon is available, the transition distance in code space modulates the mask strength:

1. Compute td = HammingDistance(mask(prev_h), mask(h)) in the 12-bit mask code.
2. Compute diff_scale = 0.5 + 0.5 * (td / 12).
3. Scale the mask deviation: mask = 1 + diff_scale * (mask - 1).

When the transition distance is small (the kernel moved a short distance in code space), the mask is attenuated toward unity. When the transition distance is large, the mask applies at full or greater strength. On the first step of a sequence, no previous horizon is available and differential modulation is not applied.

This connects the mask dynamics to the kernel's gauge structure: the transition distance td is determined by the XOR of consecutive masks, which is the same quantity that appears in the commutator and monodromy constructions.

A readable step summary:

1. **Gaussian base:** look up precomputed values from table indexed by (chi, p, distance), using CGM-grounded sigma for each chi class.
2. **Sigma-focusing:** scale the effective distance by kernel motion (large jump = narrower mask).
3. **Vertex wedge:** boost positions whose byte charge matches chi.
4. **Strength scaling:** blend toward 1 using a factor derived from w.
5. **Differential modulation:** scale mask deviation by transition distance from previous horizon.
6. **Normalization:** scale so mean(mask) = 1 and broadcast across N_feat.

Reference implementation: `compute_mask()` in `src/tools/gyrolabe.py`. Mask is computed in `GyroLabe.begin_step()` and pushed to `RoutedMLP` layers via `set_mask()`.

#### 3.4 Token advancement (model to kernel)

After sampling:

- driving_byte_t = token_id_t & 0xFF
- `kernel.step_byte(driving_byte_t)`

This is the only mechanism by which the kernel advances in the reference implementation.

Reference implementation: `advance_with_token()` in `src/tools/gyrolabe.py`.

#### 3.5 Active inference (BU-Ingress)

Before sampling from the top-k distribution, GyroLabe evaluates each candidate token by its geometric consequences for the kernel trajectory.

For each candidate token in top-k:

1. Compute b = token_id & 0xFF.
2. Peek: compute the 12-bit mask of the current horizon and XOR with the 12-bit mask of b to obtain the next horizon mask. Look up the corresponding horizon index h2.
3. Compute the Hamming distance d = code_dist(h_cur, h2) in the kernel mask code.
4. Compute the geometric cost G(b) = -log(P_code[d] + epsilon), where P_code is the kernel-native prior over code distances (the empirical histogram of all pairwise mask distances, no free parameters).
5. Adjust the logit: logit(token) = logit(token) - beta * G(b), where beta = 5/256 = A_kernel (the kernel intrinsic aperture).

Tokens that would drive the kernel to a naturally common code distance are penalized less. Tokens that would produce an unusual jump in code space are penalized more. The precision of this adjustment is set by the kernel's own aperture, not by an arbitrary hyperparameter.

This implements BU-Ingress at the token selection level: the action (token choice) is informed by where it will take the kernel trajectory, not only by the model's semantic fluency.

#### 3.6 Extraction telemetry (model to analysis only)

GyroLabe extracts telemetry from activations to measure coupling quality. This telemetry does not affect kernel stepping.

**Extraction mechanism**
The extraction of a byte from the activation peak follows a specific geometric heuristic:

1. Reshape output activation to boundary form [batch, 256, N_fiber].
2. Compute energy (squared magnitude) at each of the 256 boundary positions.
3. Identify the peak boundary index h_peak with maximal energy.
4. Select the fiber vector at that position (dimension N_fiber).
5. Form the extracted byte from the sign pattern of fiber channels: bit i is 1 if channel[2i] > 0, else 0.

**Telemetry metrics**
- **h_peak:** boundary index with maximum energy
- **peak_mass:** concentration at the peak
- **correlation:** cosine similarity between energy distribution and the applied mask
- **code_dist:** Hamming distance in mask code space between h and h_peak

**Diagnostic coordinate mu (telemetry only)**
The telemetry also computes:

- mu_t = (h_t + 64 * p_t) mod 256
- dist_to_mu = circular distance from h_peak to mu_t
- This mu_t is a diagnostic phase-lifted coordinate. It is not the center of the projection mask in the reference implementation.

Reference implementation: `extract_byte()`, `begin_step()`, `end_step()` in `src/tools/gyrolabe.py`.

#### 3.7 Code distance baseline (why "near 6" is neutral)

The kernel mask code C is a linear [12, 8] code with a palindromic weight distribution centered at weight 6. For two masks drawn without structural bias from C, the expected Hamming distance is 6 bits.

Interpretation:

- **code_dist near 6 over long trajectories** indicates the interaction preserves the intrinsic isotropy of the code space.
- **Systematic deviation away from 6** (as an average over many steps) indicates directional bias in how the coupled system explores code space.

---

### 4. Reference Implementation (files and responsibilities)

This specification corresponds to the reference implementation organized as follows:

- **Kernel stepping and observables:** `src/router/kernel.py` (uses atlas artifacts)
- **Kernel-derived constants and helpers:** `src/router/constants.py`
- **Coupling system and generation loop:** `src/tools/gyrolabe.py`
  - `GyroLabe` (wrapping, telemetry, stepping)
  - `RoutedMLP` (projection injection via received mask)
  - `compute_mask` (projection mask with Gaussian LUT and differential scaling)
  - `generate` (sampling loop with projection coupling)
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
- vertex distribution: chi histogram
- driving byte distribution: unique bytes, byte entropy
- mask weight histogram (from driving bytes)

#### 5.3 Alignment metrics

- mean_code_dist (baseline near 6.0 is neutral isotropy)
- mean_correlation (energy distribution vs mask)
- mean_gain_at_peak (mask value at peak, mask is mean-normalized to 1)
- distances to diagnostic mu (phase-lifted coordinate, telemetry only)

**Trajectory length and warm-up cost**
GyroLabe exhibits a settling period. On short generations the geometric constraints may slightly increase perplexity relative to baseline. Over longer trajectories (approximately 100 or more tokens) the joint system stabilizes, typically yielding lower perplexity, full exploration of the horizon boundary (all 256 states visited in long runs), and near-flat distribution across the four vertex classes.

**Structural observatory**
A dedicated analysis runner (scripts/run_gyrolabe_experiments.py) provides deep structural telemetry beyond the summary metrics above. It certifies kernel invariants in vivo (quotient dynamics K4 law, parity law P8, controlled BU-Egress xyxy identity) and measures windowed geometric dynamics including transport KL divergence, monodromy contraction/retraction, and palindromic complexity of kernel symbol streams.

#### 5.4 Quality metrics (logprobs and perplexity)

In the reference implementation:

- logprobs are computed from the model's logits
- sampling occurs from the standard top-k distribution

These logprobs measure "how probable were the sampled tokens under the model distribution."

---

### 6. Topological alignment

**Definition**
Topological alignment is the condition where the coupled model-kernel dynamics preserve the intrinsic metric structure of the kernel mask code C, where the intrinsic metric is **Hamming distance on the 12-bit codewords**.

Operational evidence (observed together, over long runs):

- code_dist remains near the code's natural 6-bit baseline
- correlation is consistently positive beyond null baselines
- horizon and byte entropies remain high (exploration is not collapsed)

---

### 7. Application scenarios

#### 7.1 Generative guidance

The kernel biases generation through internal coupling: a differentially modulated projection mask inside routed MLP layers. The mask strength adapts to the transition distance in code space between consecutive kernel states.

The model remains stochastic. The kernel provides a stable finite reference geometry.

#### 7.2 Distributed synchronization

Participants share a byte ledger. Replaying the kernel from the archetype through the shared ledger yields identical states, enabling coordination without trusted timestamps.

#### 7.3 Provenance and audit

A byte-encoded operation history yields a reproducible kernel trajectory and compact parity invariants. Insertions, deletions, or reorderings change the invariants.

#### 7.4 Training-time geometric calibration

GyroLabe is designed to operate at inference time on frozen model weights. However, the architecture is compatible with training-time application, where the mask would be active during forward passes followed by gradient updates.

In this configuration, gradients flow through the masked activations. Over many training steps, the model's parameters would reorganize to produce activations that work well under the mask. The 256 boundary positions would acquire geometric meaning in the learned representations. The gate/up/down topology of the SwiGLU MLP (selector, expansion, contraction) would align with the kernel's own transition structure (mask as gate, XOR as transformation, complement-and-swap as projection).

This is not yet tested. It is noted here because the mask computation, the active inference adjustment, and the kernel stepping are all differentiable-compatible or discrete-only in ways that do not obstruct gradient flow through the model's own parameters. The mask is applied as element-wise multiplication on z, which passes gradients to gate_proj and up_proj unchanged. The active inference adjustment modifies logits before sampling, which is standard in reinforcement learning from human feedback and similar training regimes.

The potential benefit of training-time application is that the model would not merely tolerate the kernel's geometry but encode it in its weights, producing native resonance rather than post-hoc calibration.

#### 7.5 Inference Function development

GyroLabe serves as an empirical laboratory for refining the principles of a standalone geometric Inference Function that could operate without a transformer backend.

The Inference Function (specified separately in the Gyroscopic ASI architecture) selects bytes by projecting a phase-aware memory field M[h, p, :] onto byte feature vectors derived from the kernel's mask geometry. It uses fixed scoring rules: feature projection, weight penalty, and vertex coherence.

GyroLabe is discovering what this Inference Function is missing:

- Dynamic coupling strength that varies with kernel motion (sigma-focus)
- Action selection informed by the kernel's transport prior (active inference via P_code)
- Settling behavior where geometric constraints and semantic fluency find joint equilibrium over time
- Contraction and retraction dynamics where the system oscillates around kernel-native measures rather than converging to a static point

Each mechanism refined in GyroLabe is a candidate for incorporation into the Inference Function. GyroLabe may ultimately become the Inference Function itself, or remain as a separate calibration layer. In either case, the experimental findings from coupling a transformer to the kernel geometry directly inform the design of systems that operate on the kernel's native computational substrate without matrix multiplication or gradient descent.

---

### 8. Analytical note

The kernel's 256-element boundary can be lifted to a 256-dimensional vector space representation. In this view, GyroLabe can be analyzed as a structural resonance between continuous model dynamics and a discrete, deterministic code geometry. Telemetry provides empirical measures of coupling strength (correlation) and geometric neutrality (code_dist baseline behavior).

**Shared computational substrate**
Both the kernel and transformer architectures are built on the same binary computational substrate. All structurally significant dimensions in both systems sit on powers of 2 (and occasionally 2^n * 3^m): the kernel's ontology (65536 = 2^16), horizon (256 = 2^8), mask code (12 bits), vertex classes (4 = 2^2), and phase (4 = 2^2) share the same arithmetic foundation as the transformer's hidden size (4096 = 2^12), layer count (32 = 2^5), head count (32 = 2^5), and position capacity (65536 = 2^16). This is not coincidence or numerology. It is a consequence of both systems being implemented in binary digital computation, which is itself constrained by physics.

The kernel makes these constraints explicit through algebraic closure conditions. The transformer satisfies them implicitly through gradient-optimized parameters. GyroLabe bridges the two by expressing the kernel's explicit geometry in the transformer's native activation space.

**Gate/up/down topology**
The SwiGLU MLP computes z = silu(gate(x)) * up(x), then output = down(z). This is a topological structure: gate selects what passes (a boundary operator), up expands into a higher-dimensional intermediate space, down contracts back to the working dimension. The kernel's transition law has the same structure: the mask selects which bits are transformed (gate), XOR applies the transformation (up), and complement-and-swap projects back and exchanges active and passive roles (down). GyroLabe currently masks the product z after gating and expansion have already occurred. The structural correspondence between the kernel's gate/up/down and the MLP's gate/up/down is noted but not yet exploited.

**Active inference as the unifying framework**
GyroLabe implements active inference with the kernel as the generative model and the transformer as the system being calibrated. The theoretical target for coupling coherence is 1 - A* = 0.9793, where A* is the CGM aperture. Current empirical correlation is in the range 0.3 to 0.5, indicating the system operates well below full coherence. This gap is expected for post-hoc coupling on frozen weights and represents the space that training-time application or a native Inference Function would close.

---

### 9. Operational considerations

#### 9.1 Batch support

The Router Kernel supports batch inference (B >= 1). All per-sequence state is stored in arrays of shape (B,):

- **Kernel state:** `state_index` and `last_byte` are numpy arrays of shape (batch_size,). Each sequence in the batch evolves an independent trajectory.
- **Atlas lookups:** Next-state transitions use numpy advanced indexing: `epistemology[state_indices, bytes]`. This is efficient on both CPU and GPU.
- **Batch management:** `resize_batch(new_size)` adjusts the batch dimension, resetting all sequences to archetype.
- **Stepping:** `step_byte()` accepts either a scalar (broadcast to all sequences) or an array of shape (batch_size,) for per-sequence bytes.

The GyroLabe coupling layer currently operates on sequence 0 of the kernel batch (single-sequence generation). The kernel's batch capability enables future multi-sequence coordination without architectural changes.

#### 9.2 Computational overhead

The coordination cycle adds minimal latency:

- **Kernel stepping:** O(1) integer array lookup per sequence.
- **Mask computation:** Computed once per token step in `begin_step()`, then reused by all routed layers. Uses precomputed Gaussian LUT (208 floats). Differential modulation adds one integer distance lookup.
- **Projection:** Element-wise multiplication on the hidden state inside each routed layer. Less than 1% of layer compute.

The dominant cost remains the model's own forward pass.

#### 9.3 Precision and determinism

**Kernel (integer domain)**
- Uses uint16/uint32 arithmetic throughout.
- Bit-exact across all hardware platforms (CPU, CUDA, MPS).
- Ledger replay produces identical states on any conforming implementation.

**Projection (floating-point domain)**
- Uses standard floating-point math (exp, division, multiplication).
- Minor numerical divergence across platforms is expected due to FP associativity and fused operations.
- This divergence does **not** affect the discrete kernel state, which depends only on the driving byte (derived from the sampled token_id, an integer).

#### 9.4 Memory considerations

**Atlas sharing**
- The atlas is read-only after loading.
- In multi-process deployments (e.g., multiple model replicas on one host), the atlas can be memory-mapped (mmap) so all processes share a single physical copy.
- Total atlas footprint: approximately 128 MiB.

**Per-sequence state**
- Each active sequence requires storage for one kernel state index (int64) plus any accumulated telemetry.
- For B concurrent sequences without stored telemetry: 16B bytes of kernel state (8 for state_index + 8 for last_byte).

#### 9.5 Failure modes

- **Missing atlas:** The kernel raises an error at construction if atlas files are not found. Generation cannot proceed.
- **Incompatible model architecture:** GyroLabe raises an error during `install()` if the model does not expose the required interface or fails divisibility checks.
- **Numeric instability:** If mask computation produces NaN or Inf (e.g., due to extreme sigma values), the reference implementation clamps or skips the mask for that step. This is logged but does not halt generation.