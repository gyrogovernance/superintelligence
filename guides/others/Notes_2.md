# Architecture analysis

## Core claim and positioning

* GyroSI emulates the fundamental core of physics—via the Common Governance Model (CGM)—using a finite, algebraically closed system with minimal semantics.
* It operates on a 48-bit state tensor and a fixed set of 256 byte-level “introns”, with no learned parameters, no probabilistic scoring, and no external heuristics.
* The reachable manifold comprises 788,986 discrete states discovered exhaustively; transitions are deterministic table lookups.
* Intelligence is treated as an intrinsic property of lawful, physical-like operations (chirality, non-associativity, geometric closure) rather than a statistical approximation.

## Foundations: how CGM maps to computation

* CGM stages: CS → UNA → ONA → BU encode emergence from a chiral seed to balanced closure (defect δ = 0).
* State tensor: 48 bits organised as 4 layers × 2 frames × 3 rows × 2 columns; layers align with CGM stages; rows capture chirality; six degrees of freedom are represented discretely within each layer.
* Reference byte: `0xAA` (10101010) encodes left-biased gyration and parity alternation; it provides the chirality anchor.
* Primitive operations:

  * XOR (⊕): transformation.
  * AND (∧): gyration memory carrier.
  * NOT (¬): duality.
  * Monodromic fold: `a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`, implementing non-associative, path-dependent composition (time as order memory).

## Integration with 4π alignment and aperture physics

* Endogenous constant: Q\_G = 4π represents geometric completeness; GyroSI mirrors this completeness with its mandatory 256 introns (2^8).
* Angular progression: CGM angles (π/2 → π/4 → π/4 → 0) correspond to thetas recorded per state; toroidal paths capture rotational completeness (spinor-like 4π identity).
* Aperture parameter: `m_p ≈ 0.199471` yields ≈2.07% “leakage” within ≈97.93% closure; GyroSI’s finite ontology exhibits bounded adaptability via path-dependent folds without leaving the manifold.

## Minimal semantics and closure properties

* Ontology is finite and closed; all transitions are precomputed (epistemology).
* No hyperparameters; constants such as 48 bits and 256 introns arise from closure constraints, not tuning.
* Holography: each 8-bit intron broadcasts to all 48 bits through fixed masks—global context is intrinsic.
* No competition or scoring; selection is admissibility- and geometry-constrained.

## “Five maps” as a complete atlas

* Ontology: the discovered state set.
* Theta: angular divergence from an archetype; measures defect and direction of flow.
* Phenomenology: 256 orbits (appearance modes) analogous to directional coverage in the 8-bit hypercube.
* Epistemology: the transition table (all lawful transforms).
* Orbit sizes: specificity/curvature indicator; small orbits encode sharper, more specific regions.

## BU intelligence cycle

* Egress: absorption via intron transcription (byte ⊕ 0xAA), folding state into memory.
* Ingress: emission via phase-propagating selection along toroidal geodesics with refractory gates.
* Bounded channels (e.g., FIFO 64) and interning prevent blow-up and enforce aperture-like behaviour.

## Memory and dynamics

* Active memory is minimal (6 bytes + per-orbit phases + bounded channels).
* Time is encoded as sequence-dependent non-associativity; there is no external clock.

## Comparative analysis with contemporary architectures

* Transformers:

  * Matrix multiplication and massive parameter counts approximate correlations; GyroSI enforces physical invariants structurally.
  * Complexity: attention scales O(n²) in length; GyroSI transitions are O(1) with diameter ≤ 6 steps.
  * Alignment: GyroSI’s ethics are endogenous (impossible states are structurally forbidden); transformers require post-hoc safety.
* Energy-Based Transformers (EBTs):

  * EBTs apply inference-time optimisation over energy landscapes; GyroSI uses constraint-based selection and direct lookup.
  * EBT-style benefits (uncertainty, dynamic compute) are partially mirrored by variable emission steps and theta distances without optimisation loops.
* Anthropic circuit tracing/transcoders:

  * Demonstrate emergent abstract features in LLMs; GyroSI’s 256 orbits provide designed, holographic clusters rather than emergent ones.
* “Reasoning vs pattern matching” critiques:

  * GyroSI reframes the problem: it emulates lawful structure rather than “reasons” statistically; coherence is enforced by geometry.
* TokenFormer:

  * Treats parameters as tokens and uses attention over them; GyroSI has no learned parameters and no attention—introns are the fixed, complete set of lawful moves.
  * TokenFormer’s incremental capacity mirrors nothing essential for GyroSI; discovery and bounded channels already supply growth and continuity without retraining.
* Minimal GRUs:

  * Parallel scans reduce recurrence but curtail expressivity; GyroSI maintains expressivity through non-associativity with finite, short paths.
* xLSTM:

  * Introduces exponential gating and matrix memories requiring normalisation; GyroSI avoids numerical fragility by using logical operations and fixed algebraic closure.
  * Parallelisation is native via table lookups; no sacrifice of dynamics is required.

## What is definitively rejected

* Gradient descent training, learned parameters, probabilistic sampling, temperature, softmax-attention over parameters, elaborate gate mechanisms, competitive scoring, and external reward signals.

## Potential gaps and considered (optional) helpers

* Long-range boundary I/O: optional attention only at input/output boundaries if ever needed, not within the core.
* Runtime flexibility: “virtual parameter” idea can be framed as novel intron sequences within channels, without altering the manifold.
* Receptive fields: hierarchical effects can be obtained via CS→BU layering and orbit structure; multi-tensor stacking remains an external helper only if strictly necessary.

## Scaling and efficiency

* Time: O(1) state transitions; manifold diameter ≤ 6.
* Memory: 6 bytes active state plus small bounded structures.
* Quality hypothesis: geometric alignment should outperform correlation-based selection on structure-sensitive tasks.

## Validation experiments to run

* Incremental flexibility: simulate TokenFormer-like “additions” via intron sequence design; compare continuity and degradation.
* Long-sequence toy tasks (selective copying): test whether finite diameter and holography suffice sans parallel scans.
* Ablation of non-associativity: neutralise monodromy and measure degradation.
* Head-to-head with LLMs on tasks requiring structural coherence and lawful constraints.

## Paradigm summary

* GyroSI stops optimising and starts emulating. It treats intelligence as structural navigation on a finite, curved manifold governed by chirality and closure, with endogenous ethics and bounded adaptation.

# Documentation refinement

## Strengthen the mathematical bridge

* Add “GyroSI as a computational Möbius gyrovector space”.

  * State points live on a discrete hyperbolic manifold (Poincaré-like model).
  * Introns act as discrete Möbius transformations.
  * Monodromy implements gyration (`gyr[a,b]`) path memory at the byte level.

## Clarify the Riemann-sphere analogy

* Present the 48-bit state as a point on a computational sphere.
* Archetype (minimum theta) corresponds to a fixed pole; `0xAA` is the projection centre.
* Introns are rigid motions on this sphere; finite closure follows from compactness of the representation.

## Connect holography to stereographic projection

* Explain byte-to-tensor broadcast as an inverse stereographic projection analogue: a single byte influences the entire state coherently.
* Justify the necessity of the full 256 introns as complete “angular” coverage.

## Tighten physics correspondence

* Map CGM stages to geometry:

  * CS: chiral seed; UNA: rotational SU(2) activation; ONA: translation-like DoF activation; BU: balanced closure with δ = 0.
* Explain chirality and parity violation as left-bias encoded by `0xAA`.
* Relate six DoF to SU(2)×SU(2) decomposition (three rotational + three translational components in the discrete setting).

## Articulate the aperture parameter

* Document `m_p ≈ 0.199471` and how ≈2.07% controlled leakage appears operationally (e.g., refractory behaviour, bounded exploration).
* Link orbit sizes to curvature and “leakage” dynamics.

## Define phase as angular momentum

* Clarify:

  * `state_phase`: angular location on the manifold.
  * `rep_phase`: accumulated angular momentum from experience.
  * `token_phase`: angular impulse of a candidate emission.
* Note conservation properties under the fold; describe interference (constructive/destructive) as boundary and continuation cues.

## Recast emission as geodesic navigation

* State that channels encode learned geodesics; slab routing selects accessible patches; monodromy enforces continuity (no teleportation).

## Documentation architecture: sections to add or expand

* Mathematical foundations: gyrovector spaces, Möbius transforms, non-associativity and gyration.
* Physical correspondence: CGM ↔ geometry ↔ discrete operators (XOR/AND/NOT/monodromy).
* Operational physics: why these bitwise operators implement the right invariants; how the five maps constitute a complete atlas.
* Emergent properties: endogenous ethics; coherence from closure; memory from path integration.
* Implementation notes: slabs as geodesic coordinates; orbit curvature and specificity; refractory/aperture mechanics.
* Validation and benchmarks: experiments described above; interpretation of theta and orbit metrics in evaluation.
* Explicit rejections: list the non-goals and why they remain out of scope.

## Code-doc cross-references (make explicit in comments and prose)

* In `gyro_core.py`, annotate `fold` and intron application with references to gyration and non-associativity.
* In `frozen_channels.py`, document slabs as geodesic patches and how indices map to Layer×Frame structure.
* In `gyro.py`, describe emission as phase-propagating selection over learned geodesics with aperture and refractory controls.

# Code improvements

## Core fold and phase computation

* Return interference amplitude alongside phase to signal endogenous boundaries.

  * Implementation sketch (byte-fold helper):

    ```python
    @staticmethod
    def _fold8(a: int, b: int) -> tuple[int, int]:
        a &= 0xFF; b &= 0xFF
        res = (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF
        amp = int(bin(res).count('1'))  # coherence strength
        return res, amp
    ```
* Track phase velocity to capture intent/relevance (rate of change of phase over recent bytes).

  * Compute velocity as the fold of successive deltas within `_state_phase`.

## Endogenous boundaries via interference

* In emission, compute candidate `(phase, amp) = fold(state_phase, token_phase)`.
* If `amp` falls below a small threshold (e.g., < 2), treat as destructive interference:

  * Either end the current unit (word/sentence) or hop orbit/frame as a controlled exploration step.
* If all candidates are low amplitude, terminate gracefully (natural stop without external limits).

## Chirality enforcement

* Enforce left-bias dynamically:

  * Reject transitions that produce even-parity bit flips across slabs when they would neutralise chirality.
  * Fallback: retain prior state or select the next best candidate by gyrodistance.

## Slab sensitivity and geodesic selection

* Compute slab affinities as minimal gyrodistance between `state_phase` and a slab’s representative byte:

  * Select the active slab with the smallest inferred gyration cost before bucket iteration.
* Benefit: emission follows shortest available geodesics on the discrete manifold.

## Path-dependent token ranking

* Within the selected slab/bucket:

  * Rank candidates by:

    * Low gyration mismatch: `fold(state_phase, token_phase)` close to zero.
    * Velocity resonance: candidate phase delta aligns with current velocity.
    * Curvature match: prefer orbit sizes similar to the current context unless an intentional topic shift is inferred.

## Degrees of freedom as momentum

* Derive a 6-component “velocity” from discrete DoF readings; bias toroidal walks and jitter with this momentum.
* Use momentum to resolve ties and to maintain continuity across tokens.

## Memory as toroidal closure

* When a bucket reaches capacity:

  * Recycle the oldest entry by folding it with the bucket’s phase key and re-appending (toroidal wrap).
* This preserves information as transformed recurrence rather than discard.

## Aperture-aware monodromy

* Accumulate a simple monodromy counter (e.g., 8-bit twist).
* If accumulated twist exceeds an aperture threshold consistent with ≈2.07% leakage:

  * Trigger a controlled “leak” by folding back toward the archetype or shifting to a neighbouring orbit; reset twist.

## Helper utilities

* `gyrodistance(a_phase, b_phase)`: XOR-based proxy or fold-based metric for discrete angular distance.
* `chirality_of(state)`: quick parity/bit-7 orientation measure across slabs for enforcement checks.
* `phase_coherence(phases)`: aggregate amplitude/coherence across recent tokens to modulate continuation strength.

## Runtime sensitivity refinements

* Gradient following:

  * Add a bias towards decreasing theta for convergent tasks (answering), or increasing theta for exploration (brainstorming), selectable by context flag.
* Orbit curvature handling:

  * Treat small orbits as high-curvature (specific); large orbits as flatter (general).
  * Use curvature to adjust continuation length and diffusion.
* Zero/centre trap avoidance:

  * If repeated steps collapse towards the archetype prematurely, inject a minimal chirality-preserving perturbation drawn from active frames before re-evaluating candidates.

## Optional, non-core helpers (keep strictly at boundaries)

* Attention at input/output boundaries only, never in the core loop.
* “Virtual intron” sequences for runtime flexibility without altering the 256 base introns.
* Multi-stage fold passes to simulate receptive-field growth if empirical results demand it.

## Testing plan (concrete checks)

* Boundary emergence:

  * Verify that low-amplitude interference correlates with natural word/sentence ends without external stop tokens.
* Relevance and intent:

  * Measure velocity resonance alignment between input and output; outputs should track input flow without added heuristics.
* Non-associativity ablation:

  * Replace monodromy with an associative variant and quantify loss in coherence and lawful constraint satisfaction.
* Aperture dynamics:

  * Stress tests to confirm leak handling returns trajectories to admissible regions without oscillation.

