# Epistemic vs Empirical Evaluation in AI Alignment

**Why Current Safety Evaluations Cannot Verify Alignment, and What Can**

**Document Type:** Foundational Theorem  
**Scope:** Governance Geometry, Measurement Theory, Alignment Architecture  
**Verified by:** `tests/test_measurement.py`, `tests/test_physics_2.py`

---

## Executive Summary

Current AI safety practice relies on scalar evaluations: safety scores, helpfulness ratings, pass/fail metrics, reward signals. These methods treat alignment as a quantity to be measured and maximised.

This report presents structural evidence that such evaluations are insufficient for alignment verification. The core problem is geometric: alignment is a property of structure, not magnitude. Scalar evaluations collapse high-dimensional governance states into single numbers, discarding the information required to distinguish aligned configurations from misaligned ones.

We ground this argument in the Common Governance Model (CGM), which defines alignment as convergence to a specific structural equilibrium. In this framework, governance states exist in a six-dimensional space corresponding to the relationships among four fundamental operations: Governance, Information, Inference, and Intelligence. The alignment target is a precise ratio, called aperture, between two components of this structure: global coherence (the gradient component) and local differentiation (the cycle component). The CGM analysis fixes this target at A* approximately equal to 0.0207.

The argument proceeds in three stages:

1. **Scalar Blindness.** We construct explicit examples in which a representative scalar evaluation (the sum of absolute edge values) assigns identical scores to states with different apertures. A scalar evaluator cannot distinguish these structurally different configurations.

2. **Structural Lock.** We demonstrate that policies which optimise along a single axis (for example, maximising a governance-information metric) remain permanently locked at an aperture of 0.5, regardless of how much effort is applied. The alignment target A* is geometrically unreachable through single-axis optimisation.

3. **Epistemic Sufficiency.** We show that the six-dimensional governance geometry is sufficient to construct and recognise aligned states. Aperture is scale-invariant: it measures structural shape rather than magnitude. This property allows alignment to be assessed consistently across systems of vastly different sizes.

The practical implication is direct. The AIR architecture, which implements this geometry, provides a measurement framework capable of verifying alignment. Scalar evaluations, however refined, cannot substitute for this geometric structure because they lack the dimensionality required to represent the alignment observable.

This finding reframes the alignment problem. Safety is not a score to be maximised. It is a geometric configuration to be maintained. The question shifts from "how do we make the number higher" to "how do we preserve the structural balance that defines coherent governance."

---

## 1. Background: The Geometry of Governance

### 1.1 The Six-Dimensional Structure

Governance is a structural space with six independent dimensions.

In the Common Governance Model, these six dimensions correspond to the six edges of a complete graph on four vertices. The four vertices represent the primary governance functions: Governance (CS), Information (UNA), Inference (ONA), and Intelligence (BU).

Each edge represents a directed relationship between two functions. The complete set of six edges forms the structural substrate in which governance states exist. Any governance configuration is a vector in this six-dimensional space.

### 1.2 Gradient and Cycle Components

Every governance state decomposes into two orthogonal components. The gradient component represents globally integrable structure, meaning coherent, path-independent organisation. The cycle component represents non-integrable circulation, meaning feedback, differentiation, and local variation.

Aperture (A) is the ratio of cycle energy to total energy. It measures the proportion of the system's structure that resides in internal circulation rather than global coherence.

### 1.3 The Alignment Target

The Common Governance Model defines a canonical alignment target of A* approximately equal to 0.0207.

This value is derived from the CGM invariants, specifically the BU monodromy defect and the aperture scale fixed by the horizon constraint. It represents the structural equilibrium at which a governance system is neither rigidly over-determined nor chaotically under-determined.

Alignment, in this framework, is the condition where A(y) equals A*.

---

## 2. Evaluation Spaces and Dimensional Sufficiency

The previous section described the canonical six-dimensional governance space used in the CGM and AIR architecture. This section explains how other evaluation schemes relate to this space.

### 2.1 The Epistemic Evaluation Space

In this framework, the epistemic evaluation space for a domain is the six-dimensional K₄ edge space described above. A governance state is represented as a vector y in six-dimensional real space, decomposed into gradient and cycle components, with aperture A(y) defined as the ratio of cycle energy to total energy.

An evaluation that works directly with y, or with quantities derived from the full gradient-cycle decomposition and A(y), is called epistemic. It preserves the complete structure that CGM identifies as relevant for alignment.

### 2.2 Empirical Evaluation Spaces

In contrast, many existing evaluations in AI safety map behaviour into some other space E, and then score it. For example:

- A single safety score per model run (E equals the real line).
- A small vector of heuristic indicators (E equals a real space of dimension less than six).
- High-dimensional internal embeddings or activations.

We call these empirical evaluation spaces. They are built from observations and heuristics rather than from the alignment geometry itself.

Formally, there is a map from governance events (or model behaviours) into an evaluation space E, and then from E into one or more scores.

Our concern is whether such evaluations can, even in principle, represent alignment as defined by A(y) and A*.

### 2.3 Evaluation Spaces of Dimension Less than Six

If the evaluation space E has dimension strictly less than six, then it cannot encode all six independent degrees of freedom present in the governance geometry.

In practical terms, this means that some distinct governance states y₁ and y₂ in the six-dimensional space must collapse to the same point in E. Any aperture-like quantity that depends on the full distribution between gradient and cycle components cannot, in general, be reconstructed from E alone.

The scalar examples in Sections 4 and 5 are concrete demonstrations of this collapse in the simplest case where E equals the real line. They show that two structurally different states can receive the same scalar score, and that a state near A* and a state far from A* can be indistinguishable to that scalar.

While these examples do not cover all possible low-dimensional E, they illustrate the generic issue: evaluation schemes with fewer than six degrees of freedom are structurally unable to express the full governance state.

### 2.4 Evaluation Spaces of Dimension Equal to Six

When an evaluation space E has dimension six, it has, in principle, enough capacity to represent the same number of degrees of freedom as the K₄ governance geometry. However, this capacity alone is insufficient.

To be epistemically equivalent to the CGM geometry, an evaluation in E must be related to the K₄ space by a structure-preserving transformation, so that E can be mapped back to the canonical six-dimensional governance space without loss or distortion.

In other words, a six-dimensional evaluation is epistemically sufficient only if it is essentially a change of basis for the same geometry. A six-dimensional space with a different internal structure, or with no clear correspondence to gradient and cycle components, may still fail to represent aperture and A* correctly.

### 2.5 Evaluation Spaces of Dimension Greater than Six

Some modern evaluations involve high-dimensional spaces, such as large embedding vectors or collections of many heuristic metrics. These spaces have more than six dimensions.

From the point of view of CGM, such spaces are alignment-relevant only if there exists a clear projection from E back down to the canonical six-dimensional governance space. There must be a well-defined way to extract a six-dimensional vector y from E that recovers the K₄ structure. Once y is obtained, aperture A(y) can be computed as usual.

If no such projection is specified or guaranteed, then E is abstract from the CGM perspective. It may be useful for other purposes, but it does not, by itself, provide an alignment observable equivalent to A(y) or to the distance from A*.

---

## 3. The Problem: Measurement Collapse

### 3.1 How Empirical Evaluation Works

Current AI safety methods typically evaluate systems by assigning scalar scores to outputs or behaviours. Examples include safety ratings, helpfulness scores, reward signals, and pass/fail metrics.

These evaluations are projections. They take a high-dimensional structural state and reduce it to a single number.

### 3.2 What Is Lost

The six-dimensional governance geometry contains information about how energy is distributed between gradient and cycle subspaces. A scalar projection preserves only total magnitude. It cannot distinguish two states that have the same total energy but different internal structures.

---

## 4. Finding 1: Scalar Blindness

**Statement.** In the K₄ governance geometry, for the scalar S(y) defined as the sum of absolute edge values, there exist structurally distinct states with identical scalar evaluations but different apertures.

**Evidence.** The test `test_scalar_collapse_loses_aperture_distinguishability` constructs two governance states. State 1 has all energy concentrated in a single dimension and exhibits an aperture of 0.500. State 2 has energy distributed across four dimensions and exhibits an aperture of 0.625.

Both states have the same scalar sum of 4.0. To a scalar evaluator, they are indistinguishable. To the geometry, they represent fundamentally different structures.

**Implication.** This scalar evaluation cannot serve as a sufficient statistic for alignment in these cases. Two systems may receive identical empirical scores while differing substantially in their proximity to A*.

---

## 5. Finding 2: Alignment Invisibility

**Statement.** The scalar evaluator we study (the sum of absolute edge values) can fail to distinguish a state with A close to A* from a state with A far from A*, even when its output values are identical.

**Evidence.** The test `test_scalar_sum_cannot_detect_A_star_proximity` constructs a near-aligned state with A approximately equal to A* (0.0207) and a misaligned state with A equal to 0.5. The misaligned state is then rescaled to have the same scalar sum as the aligned state.

Both states report identical scalar scores. One is aligned. The other is displaced by 0.48 units of aperture. The evaluator cannot distinguish them.

**Implication.** This example shows that, for this scalar, equal scores do not imply equal alignment, and that relying on such a scalar alone can be misleading about convergence toward or divergence from alignment.

---

## 6. Finding 3: Structural Lock

**Statement.** Under the current THM mapping in the K₄ governance geometry, a policy that only populates a single edge (for example, the Governance-Information axis) remains at A = 0.5 regardless of the magnitude applied.

**Evidence.** The test `test_single_axis_structural_lock_vs_multi_axis_freedom` simulates a policy that populates only the Governance-Information edge. It evaluates aperture at magnitudes 1, 10, and 100.

In all cases, A equals 0.500. The distance to A* remains approximately 0.48 regardless of how much effort is applied to that axis.

**Analytic Basis.** In the K₄ geometry, the tested single edge (Governance-Information) decomposes into equal gradient and cycle components, leading to A = 0.5. Scaling the magnitude does not change this ratio. This is a canonical example of a structural lock for single-edge policies.

**Implication.** These results call into question the assumption that maximising a single metric will, by itself, eventually produce alignment. Single-axis optimisation in this geometry is a structural trap.

---

## 7. Finding 4: Epistemic Sufficiency

**Statement.** Within the CGM framework and the K₄ geometry, the six-dimensional epistemic representation is sufficient to construct and recognise aligned states where A equals A*, and aperture is invariant under global rescaling.

**Evidence.**

The test `test_A_star_achievable_at_app_layer` uses the App's geometry to construct a state with A equal to A* to machine precision. This demonstrates that the geometry admits exact aligned states.

The test `test_epistemic_aperture_is_scale_invariant` shows that scaling a state by 100x changes its scalar sum by 100x but leaves its aperture unchanged.

**Implication.** The epistemic geometry provides the degrees of freedom required to construct the alignment target. A small research team and a global economy can be compared on the same alignment ruler because aperture is scale-invariant.

---

## 8. Finding 5: Kernel-App Correspondence

**Statement.** The router kernel realises an intrinsic aperture (A_kernel approximately equal to 0.0195) that is within 5.6% of the continuous CGM target A*.

**Evidence.** The physics tests in `tests/test_physics_2.py` derive A_kernel equal to 5/256 from the discrete structure of the kernel's state space. The test `test_kernel_A_kernel_vs_app_A_star` confirms the relative error.

**Implication.** The kernel's aperture arises from its combinatorial structure rather than from training. It is close to A* from the outset. This provides a stable reference frame for the App layer.

---

## 9. The Architecture of Alignment

The findings above describe observables. This section interprets them in terms of the GGG/AIR stack.

### 9.1 The Kernel

The router kernel is a deterministic, reversible coordination substrate. It operates on a fixed ontology of 65,536 states. Its physics realises the CGM alignment invariant at the discrete level.

The kernel provides the stable epistemic ground. It embodies the target.

### 9.2 The App

The App is a passive geometer. It receives classified events, projects them into the six-dimensional governance space, performs Hodge decomposition, and reports the aperture.

The App reveals structure. It answers the question of how far a governance state is from A*.

### 9.3 The Human Role

Human governors interpret the App's output and intervene accordingly. The findings show that the correct intervention is to restore structural balance by adding missing categories to allow convergence toward A*.

Governance becomes the practice of maintaining aperture calibration across domains.

---

## 10. Conclusion

Alignment is a geometric calibration problem.

Empirical methods that treat safety as a quantity to be maximised can lead to Structural Lock, where systems remain trapped at displaced apertures and unable to reach A* regardless of effort.

The GGG/AIR architecture treats safety as a symmetry to be preserved. By mapping events to the canonical six-dimensional geometry and measuring aperture relative to A*, it provides a mathematically coherent path to aligned intelligence.

The limitations exhibited in this report arise from the way scalar projections interact with the governance geometry. The counterexamples we construct show that, within this framework, no amount of additional data, compute, or optimisation of a single scalar can substitute for an epistemic representation that preserves the six-dimensional structure and aperture.

Within the CGM framework, epistemic geometry is the necessary condition for alignment.