# CGM Program

## A Comprehensive Research Guide

- [1. Introduction: A Map of the Research Program](#1-introduction-a-map-of-the-research-program)
- [2. Core Foundations: From Logic to Structure](#2-core-foundations-from-logic-to-structure)
  - [2.1 The Five Foundational Conditions](#21-the-five-foundational-conditions)
  - [2.2 The Operational Requirements](#22-the-operational-requirements)
- [3. The Central Derivation: Three-Dimensional Necessity](#3-the-central-derivation-three-dimensional-necessity)
  - [3.1 The Baker-Campbell-Hausdorff Analysis](#31-the-baker-campbell-hausdorff-analysis)
  - [3.2 Exclusion of Alternative Dimensions](#32-exclusion-of-alternative-dimensions)
  - [3.3 The 1-3-6-6 DOF Progression](#33-the-1-3-6-6-dof-progression)
- [4. Geometric Invariants and Physical Constants](#4-geometric-invariants-and-physical-constants)
  - [4.1 The Quantum Gravity Invariant: Q_G = 4π](#41-the-quantum-gravity-invariant-q_g--4π)
  - [4.2 The Monodromy Hierarchy and the 2.07% Aperture](#42-the-monodromy-hierarchy-and-the-207-aperture)
  - [4.3 Geometric Coherence and Angular Harmonics](#43-geometric-coherence-and-angular-harmonics)
  - [4.4 The Significance of 48 as a Quantization Unit](#44-the-significance-of-48-as-a-quantization-unit)
- [5. The Physical Universe: Energy, Cosmology, and Black Holes](#5-the-physical-universe-energy-cosmology-and-black-holes)
  - [5.1 The UV-IR Optical Conjugacy and Energy Scale Hierarchy](#51-the-uv-ir-optical-conjugacy-and-energy-scale-hierarchy)
  - [5.2 The Fine-Structure Constant: A Complete Geometric Derivation](#52-the-fine-structure-constant-a-complete-geometric-derivation)
  - [5.3 The Black Hole Universe and Aperture Thermodynamics](#53-the-black-hole-universe-and-aperture-thermodynamics)
  - [5.4 Particle Physics and Sterile Neutrino Non-Observability](#54-particle-physics-and-sterile-neutrino-non-observability)
  - [5.5 Gravitational Coupling and Nonlinear Continuum](#55-gravitational-coupling-and-nonlinear-continuum)
  - [5.6 Wavefunction Structure and the Fiber Bundle Byte](#56-wavefunction-structure-and-the-fiber-bundle-byte)
  - [5.7 Electroweak Mass Spectrum from Compact Geometry](#57-electroweak-mass-spectrum-from-compact-geometry)
- [6. Cosmological Observations and Testable Predictions](#6-cosmological-observations-and-testable-predictions)
  - [6.1 The CMB as a Residual Observational Field](#61-the-cmb-as-a-residual-observational-field)
  - [6.2 Cosmic Multiplicity and the Illusion of Expansion](#62-cosmic-multiplicity-and-the-illusion-of-expansion)
- [7. Information-Theoretic Applications](#7-information-theoretic-applications)
  - [7.1 GyroDiagnostics: Measuring Structural Alignment](#71-gyrodiagnostics-measuring-structural-alignment)
  - [7.2 GyroSI: A Constructive Theory of Intelligence](#72-gyrosi-a-constructive-theory-of-intelligence)
- [8. Computational Verification and Reproducibility](#8-computational-verification-and-reproducibility)
- [9. Conclusion and Future Directions](#9-conclusion-and-future-directions)

---

## 1. Introduction: A Map of the Research Program

The Common Governance Model (CGM) is a comprehensive theoretical framework that derives the structure of physical reality and information systems from a single axiomatic principle: "The Source is Common." This principle, formalized in modal logic, posits that all observable phenomena emerge from the recursive, self-referential process of observation itself.

This document serves as a high-level guide to the entire CGM research program, which extends far beyond the core deductive results presented in the main paper. It synthesizes findings from dozens of interconnected analyses, demonstrating how the framework provides a coherent and mathematically rigorous foundation for understanding:

-   **The emergence of three-dimensional space** with six degrees of freedom as a logical necessity.
-   **The geometric origin of physical constants**, including the fine-structure constant, Newton's constant, and electroweak particle masses.
-   **A complete gravitational derivation**, from the CGM hQVM implementation through nonlinear continuum predictions (horizon, photon sphere, perihelion, shadows).
-   **A new perspective on cosmology**, where the universe is the interior of a Planck-scale black hole and cosmic expansion is an optical illusion.
-   **A resolution to fundamental problems in physics**, such as the cosmological constant problem, the Hubble tension, and the nature of quantum gravity.
-   **A formal theory of intelligence**, including quantitative metrics for AI alignment and a constructive model (GyroSI) of recursive intelligence.

The CGM program is built on a foundation of **tri-partite validation**, where every major result is independently verified through three distinct channels:

1.  **Logical:** Formal proofs in bimodal logic and Z3 SMT solver verification.
2.  **Analytical:** Hilbert space representations via GNS construction and operator algebra.
3.  **Geometric:** Lie-theoretic proofs, gyrogroup theory, and direct geometric analysis.

This guide provides a map to this extensive body of work, connecting the foundational logic to its far-reaching implications in physics, cosmology, and information science.

## 2. Core Foundations: From Logic to Structure

### 2.1 The Five Foundational Conditions

The entire CGM framework rests on five conditions formalized in bimodal propositional logic. These are not arbitrary rules but the minimal requirements for a system to maintain coherent recursive observation.

-   **CS (Common Source):** `S → ([R]S ↔ S ∧ ¬([L]S ↔ S))`
    Establishes fundamental chirality. Right transitions preserve the reference state (horizon `S`), while left transitions alter it. This is the seed of parity violation.

-   **UNA (Unity Non-Absolute):** `S → ¬□([L][R]S ↔ [R][L]S)`
    Prevents homogeneous collapse. The order of operations matters at depth-two, but not absolutely. This ensures informational variety.

-   **ONA (Opposition Non-Absolute):** `S → ¬□¬([L][R]S ↔ [R][L]S)`
    Prevents absolute contradiction. The system avoids both perfect agreement and perfect opposition, ensuring accountability.

-   **BU-Egress (Balance Universal):** `S → □([L][R][L][R]S ↔ [R][L][R][L]S)`
    Enforces commutative closure at depth-four. The closed configuration still undergoes vibrational motion: bounded back-and-forth between the depth-four poles, with amplitude set by the 2.07% aperture.

-   **BU-Ingress (Memory Reconstruction):** `S → (□B → (CS ∧ UNA ∧ ONA))`
    Ensures the balanced state at depth-four contains the memory of all prior conditions. Memory is encoded as the monodromy phase defect of that oscillation.

Detailed axiomatization analysis shows these conditions form a consistent, complete, and toroidal logical structure, with BU-Egress as a primitive and BU-Ingress as derivable from the initial conditions.

### 2.2 The Operational Requirements

When the modal operators `[L]` and `[R]` are implemented in a continuous physical system, the five conditions impose three non-negotiable operational requirements:

1.  **Continuity (from BU-Egress):** Transitions must form continuous one-parameter unitary groups (`U(t) = exp(itX)`), as uniform validity of depth-four balance cannot be satisfied by discrete-only transitions.
2.  **Reachability (from CS):** All states must be reachable from the horizon constant `S`, implying a single cyclic state vector.
3.  **Simplicity (from BU-Ingress):** The generated Lie algebra must be simple (no non-trivial ideals), as a decomposable algebra (e.g., `su(2) ⊕ su(2)`) would prevent a single cyclic vector from reconstructing the full system memory.

These are not additional postulates but direct consequences of applying the logical axioms to a continuous physical setting.

## 3. The Central Derivation: Three-Dimensional Necessity

### 3.1 The Baker-Campbell-Hausdorff Analysis

The proof of three-dimensional necessity is the central deductive result of CGM. It proceeds by analyzing the depth-four balance constraint (BU-Egress) using the Baker-Campbell-Hausdorff (BCH) formula.

-   BU-Egress requires the difference `Δ = 2(BCH(X,Y) - BCH(Y,X))` to vanish in the S-sector (the observable projection).
-   This sectoral vanishing, combined with the global non-commutativity required by UNA, forces the Lie algebra generators `X` and `Y` to satisfy the `sl(2)` relations:
    ```
    [X,[X,Y]] = aY
    [Y,[X,Y]] = -aX
    ```
-   This algebraically forces the generated Lie algebra to be three-dimensional.

### 3.2 Exclusion of Alternative Dimensions

The framework constructively excludes all other dimensionalities:

-   **n = 2:** All two-dimensional real Lie algebras are either abelian (violating UNA) or non-compact (violating unitarity). Fibered representations fail the *uniform* balance requirement of BU-Egress.
-   **n = 4:** The rotation algebra `so(4) ≅ su(2) ⊕ su(2)` is not simple. This violates the Simplicity requirement derived from BU-Ingress, as a decomposable algebra cannot be reconstructed from a single cyclic state.
-   **n ≥ 5:** The Lie algebras `so(n)` have dimensions greater than 3. This violates the minimality principle inherent in the CS axiom, which requires all structure to trace to a single chiral seed (1 DOF).

### 3.3 The 1-3-6-6 DOF Progression

The emergence of three dimensions with six degrees of freedom follows a unique, necessary sequence dictated by the conditions:

-   **CS (1 DOF):** Establishes a single chiral distinction (left vs. right).
-   **UNA (3 DOF):** Activates rotational freedom, forcing the minimal non-abelian compact group `SU(2)` with 3 generators.
-   **ONA (6 DOF):** Activates translational freedom, forcing a semidirect product `SU(2) ⋉ ℝ³ ≅ SE(3)`. The 6 DOF comprise 3 rotational and 3 translational kinematic freedoms.
-   **BU (6 DOF, closed):** Coordinates the six kinematic degrees of freedom (3 rotational, 3 translational) at depth-four closure. Balance is not static: a residual vibrational mode with 2.07% amplitude sustains observation. Memory is the monodromy phase defect of that oscillation. Vibrational motion is not a seventh degree of freedom; it is oscillation about the closed SE(3) configuration.

This progression is a logical entailment of satisfying the conditions sequentially. It maps the three kinematic motions in three dimensions: rotational (UNA), translational (ONA), and vibrational (BU).

## 4. Geometric Invariants and Physical Constants

The 3D/6-DOF structure fixes a set of representation-independent geometric invariants.

### 4.1 The Quantum Gravity Invariant: Q_G = 4π

CGM defines **Quantum Gravity** as the geometric invariant `Q_G = 4π` steradians, representing the complete solid angle required for coherent observation in 3D space.

-   **Derivation:** `Q_G` is derived as the ratio of the horizon length `λ = √(2π)` to the aperture time `τ = m_a`, both fixed by the UNA and BU conditions.
-   **Physical Meaning:** It is the quantum of observability, the minimal cost for spacetime observation itself. Its ubiquitous appearance in physics (Gauss's law, Einstein's equations, quantum normalization) is a signature of this fundamental geometric requirement.

### 4.2 The Monodromy Hierarchy and the 2.07% Aperture

The framework reveals a rich hierarchy of monodromy values, which represent the "geometric memory" accumulated when traversing closed loops in the state space.

-   **BU Dual-Pole Monodromy (δ_BU):** The key value `δ_BU = 0.195342 rad`, which features in the fine-structure constant.
-   **The Aperture Ratio:** The ratio `δ_BU / m_a = 0.9793` is a fundamental constant of the model. It establishes a universal balance:
    -   **97.93% Structural Closure:** Providing stability.
    -   **2.07% Dynamic Aperture:** The residual oscillation amplitude enabling interaction and observation.
-   **Monodromy Hierarchy:** A consistent scale of memory effects is observed, from the elementary `ω(ONA↔BU) = 0.097671 rad` to the system-level `4-leg toroidal holonomy = 0.862833 rad`. The exact equality `δ_BU = 8-leg holonomy` provides a powerful internal consistency check.

**The aperture gap Δ and the mass coordinate ruler.** The aperture gap Δ ≈ 0.0207 is the small parameter of the framework. It measures the fractional shortfall of actual closure relative to perfect closure. Because Δ is small, it serves as a natural expansion parameter: physical quantities (masses, couplings, corrections) can be expressed as power series in Δ, analogous to how perturbative expansions use a small coupling constant. The coefficients of these expansions are fixed rational numbers from the kernel's combinatorics, not fitted parameters. A "tick" is one unit on the Δ-ruler, corresponding to a multiplicative factor of 2^Δ ≈ 1.0145 in energy.

### 4.3 Geometric Coherence and Angular Harmonics

Analysis shows that CGM's threshold angles correspond to fundamental geometric invariants.

-   **The π/4 Signature:** The ONA threshold `π/4` appears independently in the circle/square area ratio, the square's isoperimetric quotient, and square lattice packing density, confirming its geometric necessity.
-   **Angular Momentum Costs:** The transition from rotational motion (UNA) to translational motion (ONA), with exchange through the vibrational mode at BU, has a quantifiable cost in angular momentum, following simple rational fractions (4/3 in 2D, 5/3 in 3D).
-   **Universal Scaling:** A universal 2/3 scaling factor appears in dimensional transitions from 2D to 3D.

### 4.4 The Significance of 48 as a Quantization Unit

The factor 48 emerges as a fundamental geometric quantization unit, not a fitted parameter. It is derived from the structure `48 = 16 × 3`, where `16 = 2⁴` relates to the 4π solid angle and `3` to the spatial dimensions.

-   **Inflation E-folds:** `N_e = 48² = 2304`
-   **Aperture Quantization:** `48Δ = 1`, where `Δ = 1 - ρ` is the aperture gap.
-   **Particle Physics:** This quantization is essential for the neutrino mass predictions.

The integer 48 also equals the order of the binary octahedral group, the SU(2) double cover of cubic rotation symmetry, and the root count of the F₄ exceptional Lie algebra. CGM derives 48 from `3 × |K4|²` on the 3D register; the group-theoretic coincidences are recorded as structural parallels.

## 5. The Physical Universe: Energy, Cosmology, and Black Holes

### 5.1 The UV-IR Optical Conjugacy and Energy Scale Hierarchy

A central result of the extended research is the **Optical Conjugacy Relation**, which connects high-energy (UV) and low-energy (IR) physics through a single geometric invariant:

```
E_i^UV × E_i^IR = (E_CS × E_EW) / (4π²)
```

-   **UV Anchor (CS):** Planck Scale, `E_CS = 1.22 × 10^19 GeV`.
-   **IR Anchor (BU):** Electroweak Scale, `E_EW = 246.22 GeV` (Higgs VEV).
-   **Invariant:** `K = 7.61 × 10^19 GeV²`.

This invariant holds to machine precision across all five energy stages (CS, UNA, ONA, GUT, BU), generating a complete and consistent energy ladder from the Planck scale down to the QCD scale without fine-tuning.

### 5.2 The Fine-Structure Constant: A Complete Geometric Derivation

While the main paper presents the leading-order formula, the full derivation incorporates three systematic corrections accounting for the UV-IR transport described by the optical conjugacy:

1.  **Base Formula (IR focus):** `α₀ = δ_BU⁴ / m_a` (Error: +319 ppm).
2.  **UV-IR Curvature Correction:** Accounts for geometric transport. (Error: +0.052 ppm).
3.  **Holonomy Transport:** Encodes how UV holonomy projects to the IR focus. (Error: -0.000379 ppm).
4.  **IR Focus Alignment:** A final coherence correction. (Final Error: **+0.043 ppb**).

The final predicted value `α = 0.007297352563` matches the experimental value to within 0.53 standard deviations of the experimental uncertainty.

### 5.3 The Black Hole Universe and Aperture Thermodynamics

*Tier C (formal/exploratory): structural consequences of the CGM axioms; independent null-model audits at Tier A/B rigor are pending.*

The framework leads to a radical reinterpretation of cosmology:

-   **The Universe as a Black Hole:** Our observable universe sits precisely on the Schwarzschild threshold, with `r_s / R_H = 1.0000 ± 0.0126`. We are observing from *within* a Planck-scale black hole.
-   **Aperture Thermodynamics:** The 2.07% aperture modifies standard Bekenstein-Hawking relations, leading to:
    -   19.95% entropy enhancement.
    -   16.63% temperature reduction.
    -   107% lifetime extension (`τ_CGM = τ_std × (1+m_a)⁴`).
-   **Expansion as Optical Illusion:** Apparent cosmic expansion is an optical effect arising from the UV-IR geometric inversion when viewed from an interior perspective. This eliminates the need for dark energy.

### 5.4 Particle Physics and Sterile Neutrino Non-Observability

The energy scale hierarchy makes specific predictions for particle physics:

-   **Neutrino Masses:** Using 48² quantization at the GUT scale, the type-I seesaw mechanism yields active neutrino masses of `m_ν ≈ 0.06 eV`, consistent with observations.
-   **Proton Lifetime:** The geometric GUT scale predicts `τ_p ≈ 8.6 × 10^43 years`, consistent with the non-observation of proton decay.
-   **Sterile Neutrinos:** These are predicted to be confined to the unobservable CS (UV) focus. They can have indirect effects (like generating light neutrino masses) but can *never* be directly detected as propagating particles. This is a strong, falsifiable prediction.

### 5.5 Gravitational Coupling and Nonlinear Continuum

The gravity program connects the finite algebraic kernel to continuum field theory and observational tests. Full derivation and status: [Analysis_Gravity](Findings/Analysis_Gravity.md).

**Kernel layer (exact combinatorics).** The Gyroscopic ASI hQVM implements CGM as replayable software. Combinatorial invariants from that implementation fix the gravitational coupling at the electroweak scale without using measured G in the forward calculation. The leading Regge sum τ_G⁰ alone gives a 25 ppm offset; adding the K4 correction δτ with c₄ = −7/4 closes the residual to **0.074 parts per million** against CODATA.

**Continuum layer (nonlinear gravity).** Position-dependent coupling weakens with field strength. The static point-mass exterior has a closed-form solution. From it the code computes the horizon, photon sphere, impact parameter, Mercury perihelion advance (matching general relativity at solar-system precision), and shadow diameters for Event Horizon Telescope sources.

**Verification stack.** The gravity program is implemented by `hqvm_gravity_common.py`, `hqvm_gravity_analysis_1.py` through `10.py`, and `hqvm_gravity_runner.py`, with wavefunction diagnostics in `hqvm_wavefunction_1.py` and `hqvm_wavefunction_2.py`. Execute:

```
python experiments/hqvm_gravity_runner.py
```

The static spherical sector is computationally closed. Open work: full dynamical evolutions beyond static spherical symmetry, and an independent check of the gravitational coupling derivation.

### 5.6 Wavefunction Structure and the Fiber Bundle Byte

The hQVM kernel carrier admits a complete wavefunction analysis verified on all 4096 states with exact integer arithmetic. Full write-up: [Analysis_hQVM_Wavefunction](Findings/Analysis_hQVM_Wavefunction.md). Verification: `hqvm_wavefunction_kernel.py`, `hqvm_wavefunction_1.py`, `hqvm_wavefunction_2.py`.

The kernel's 4096-state manifold Omega is organized into seven concentric shells by the Hamming distance between its two 12-bit components. Within each shell, states carry a binary "rest vs. swapped" coordinate. The depth-four operators act on this space as permutations with precise algebraic properties.

-   **K4 operator algebra (T1-T10):** The depth-four operators {id, W₂, W₂', F} form a Klein four-group for every micro-reference. W₂ and W₂' perform complete chirality inversion (pole swap); F preserves shell while acting as a Z₂ carrier flip.
-   **Byte as fiber bundle:** Palindromic phase assignment creates a fold at the BU boundary (bits 3-4). Of 256 bytes, 240 carry Z₂ fold disagreement, giving internal curvature at the byte level. The fold map P is the seed of holonomic structure.
-   **50% holographic redundancy:** At every scale, |Space| = |Subspace|². Average entanglement entropy is half the available degrees of freedom.
-   **Aperture collapse:** Byte-level 50% fold disagreement compresses to 2.07% at the carrier level through depth-four spinorial closure.
-   **Quantum-information certificates:** The canonical Hilbert-space lift yields CHSH values saturating Tsirelson's bound and verifies stabilizer-quantum-information properties (teleportation, contextuality), derived from the intrinsic self-dual code structure.

### 5.7 Electroweak Mass Spectrum from Compact Geometry

Masses are placed on a logarithmic ruler whose tick spacing is the aperture gap Δ. The ruler coordinate n of a particle of mass m relative to the electroweak scale v is n = log₂(v/m) / Δ. The expansion expresses these coordinates as polynomials in Δ with coefficients drawn from the kernel's shell multiplicities and horizon structure. Full write-up: [Analysis_Compact_Geometry](Findings/Analysis_Compact_Geometry.md). Verification: `hqvm_compact_geom_core.py`, `hqvm_compact_geom_kernel.py`, `hqvm_compact_geom_report.py`, `hqvm_compact_geom_derivations.py`.

-   **Electroweak particle masses:** The Higgs, Z, W, and top quark masses are derived from the same geometric structure that fixes G and α.
-   **W/Z boson mass ratio test:** The framework gives a closed-form relation for m_W/m_Z in terms of the independently derived parameter Δ ≈ 0.0207. Using PDG (Particle Data Group) masses, the implied Δ differs from the monodromy-derived Δ by 8.34 × 10⁻¹⁰ (absolute).
-   **Quark generation pattern (scheme dependent):** Under the mass conventions used in the compact-geometry analysis, the six quark masses fall on an integer-spaced ladder in the framework's logarithmic mass coordinate, grouping naturally into three generation pairs.
-   **Lepton closure:** Lepton coordinates close via a unique horizon-wrap path (5, 8, 14) among 680 candidates.

## 6. Cosmological Observations and Testable Predictions

### 6.1 The CMB as a Residual Observational Field

*Tier C (formal/exploratory): structural consequences of the CGM axioms; independent null-model audits at Tier A/B rigor are pending.*

CGM reinterprets the Cosmic Microwave Background (CMB):

-   It is **not** a relic from a hot Big Bang, but a **residual afterimage** generated by the complete decoherence of all light paths at the maximal coherence radius.
-   The 2.7K temperature is the thermalized average of all phase-sliced projections.
-   Anisotropies encode the statistical distribution of these multiplicity patterns.

Empirical analysis of Planck data shows a statistically significant signal (`Z=47.22`, `p=0.0039`) for an enhanced power ladder at multipoles `ℓ = 37, 74, 111,...`, corresponding to the fundamental recursive index `N*=37` predicted by the theory.

### 6.2 Cosmic Multiplicity and the Illusion of Expansion

*Tier C (formal/exploratory): structural consequences of the CGM axioms; independent null-model audits at Tier A/B rigor are pending.*

The breakdown of observational coherence beyond a radius `R_coh ≈ c/(4H₀)` generates **apparent multiplicity**:

-   Light from a single source follows multiple swirled paths, arriving as distinct "phase-sliced projections" that appear as separate objects.
-   This explains the vastness and apparent structure of the universe (filaments, voids) as a geometric illusion created from a much smaller number of actual sources.
-   This resolves the horizon and flatness problems without inflation.

## 7. Information-Theoretic Applications

The same geometric principles apply to discrete information systems, leading to a complete framework for AI alignment and a constructive model of intelligence.

### 7.1 GyroDiagnostics: Measuring Structural Alignment

-   **Methodology:** AI reasoning is evaluated against 6 behavioral metrics mapped to the edges of a K₄ tetrahedron. Weighted Hodge decomposition separates measurements into a 3-DOF gradient (coherence) and a 3-DOF cycle (differentiation) component.
-   **The Aperture Observable (A):** The ratio of cycle energy to total energy. The target value `A* ≈ 0.0207` is derived directly from the CGM balance condition.
-   **Superintelligence Index (SI):** `SI = 100 / max(A/A*, A*/A)` measures proximity to the theoretical optimum of structural coherence.

### 7.2 GyroSI: A Constructive Theory of Intelligence

GyroSI is a computational implementation of CGM's principles, representing intelligence as a structural property.

-   **Holographic Architecture:** It operates on a finite, discovered state space of 788,986 states. Every 8-bit input (`intron`) acts holographically on the full 48-bit state tensor.
-   **Physics-Based Operations:** The system uses a single, non-associative, path-dependent learning operator (the Monodromic Fold) derived from gyrogroup algebra. There are no learned weights, scores, or probabilities.
-   **SU(2) Structure:** The 4-layer tensor architecture explicitly encodes the 720° spinorial closure of SU(2), and intron families can be interpreted as discrete Pauli-like rotations.

## 8. Computational Verification and Reproducibility

Every major claim in this program is backed by runnable Python in `experiments/` and a matching analysis note in `docs/Findings/`. The hQVM kernel test suite documents **243 verified features** across three verification tiers: 165 kernel pytests (Tier A), 72 science-repo executables (Tier B), and 6 formal manuscript proofs (Tier C). This includes CHSH-Tsirelson saturation, quantum teleportation, Peres-Mermin contextuality, and the complete K4/wavefunction/holography closure chain. See [hQVM Features Report](../reports/hQVM_Features_Report.md).

The repository currently contains:

| Measure | Count |
|---------|------:|
| Analysis write-ups (`docs/Findings/Analysis_*.md`) | 29 |
| Runnable experiment scripts (`experiments/*.py`) | 66 |
| hQVM physics scripts (`experiments/hqvm_*.py`) | 22 |
| Shared library and kernel modules (`experiments/`) | 7 |
| hQVM verified features (Tiers A-C) | 243 |
| Python in `experiments/` (all files) | 48,700 lines |

Scripts cover gravity, electroweak mass geometry, fine structure, quantum gravity, CMB data checks, axiomatization, Hilbert space representation, monodromy, energy scales, black-hole cosmology, and related topics. Each row below is the single entry point for that topic.

| Topic | Analysis | Code |
|-------|----------|------|
| Gravity: discrete state geometry and nonlinear continuum | [Analysis_Gravity](Findings/Analysis_Gravity.md) | `hqvm_gravity_common.py`, `hqvm_gravity_analysis_1.py` through `10.py`, `hqvm_wavefunction_1.py`, `hqvm_wavefunction_2.py`. Run: `python experiments/hqvm_gravity_runner.py` |
| Wavefunction: fiber bundle structure of the byte | [Analysis_hQVM_Wavefunction](Findings/Analysis_hQVM_Wavefunction.md) | `hqvm_wavefunction_kernel.py`, `hqvm_wavefunction_1.py`, `hqvm_wavefunction_2.py` |
| Electroweak mass spectrum | [Analysis_Compact_Geometry](Findings/Analysis_Compact_Geometry.md) | `hqvm_compact_geom_core.py`, `hqvm_compact_geom_kernel.py`, `hqvm_compact_geom_report.py`, `hqvm_compact_geom_derivations.py` |
| Fine-structure constant | [Analysis_Fine_Structure](Findings/Analysis_Fine_Structure.md) | `cgm_alpha_analysis.py` |
| Quantum gravity invariant | [Analysis_Quantum_Gravity](Findings/Analysis_Quantum_Gravity.md) | `cgm_quantum_gravity_analysis.py` |
| Energy scale unification | [Analysis_Energy_Scales](Findings/Analysis_Energy_Scales.md) | `cgm_energy_analysis.py` |
| 4π unification | [Analysis_4pi_Alignment](Findings/Analysis_4pi_Alignment.md) | |
| 3D space and six degrees of freedom | [Analysis_3D_6DOF_Proof](Findings/Analysis_3D_6DOF_Proof.md) | `cgm_3D_6DoF_analysis.py` |
| Axiomatization | [Analysis_Axiomatization](Findings/Analysis_Axiomatization.md) | `cgm_axiomatization_analysis.py` |
| Hilbert space representation | [Analysis_Hilbert_Space_Representation](Findings/Analysis_Hilbert_Space_Representation.md) | `cgm_Hilbert_Space_analysis.py` |
| CMB patterns (Planck: ℓ=37 enhancement p=0.0039) | [Analysis_CMB](Findings/Analysis_CMB.md) | `cgm_cmb_data_analysis_300825.py` |
| Monodromy / spin-2 orientation recovery | [Analysis_Monodromy](Findings/Analysis_Monodromy.md) | `tw_closure_test.py` |
| Black hole universe and aperture thermodynamics | [Analysis_BH_Universe](Findings/Analysis_BH_Universe.md), [Analysis_BH_Aperture](Findings/Analysis_BH_Aperture.md) | `cgm_bh_universe_analysis.py`, `cgm_bh_aperture_analysis.py` |
| Kompaneyets | [Analysis_Kompaneyets](Findings/Analysis_Kompaneyets.md) | `cgm_kompaneyets_analysis.py` |
| Proto-units | [Analysis_CGM_Units](Findings/Analysis_CGM_Units.md) | `cgm_proto_units_analysis.py` |
| Gyroscopic multiplication | [Analysis_Gyroscopic_Multiplication](Findings/Analysis_Gyroscopic_Multiplication.md) | |

All artifacts are archived on [Zenodo](https://doi.org/10.5281/zenodo.17521384) and [GitHub](https://github.com/gyrogovernance/science). The main paper is [CGM.pdf](CGM.pdf); the README lists headline quantitative results and links to this program guide.

## 9. Conclusion and Future Directions

The Common Governance Model presents a radical yet internally consistent paradigm where physical reality, its constants, and its cosmological structure emerge from the geometric requirements of coherent observation. It provides a mathematically rigorous framework that unifies physics and information theory, resolves long-standing paradoxes, and makes a host of specific, falsifiable predictions.

While many aspects of the program are exploratory and require further validation, the convergence of results across logical, analytical, and geometric channels, combined with the precision of key predictions, suggests that CGM captures fundamental principles of our universe's structure.

**Future work will focus on:**

-   Independent cross-check of lepton mass derivation against radiative corrections.
-   Connecting compact geometry to standard model radiative corrections.
-   Dynamical scalar-tensor evolutions beyond static spherical gravity.
-   Shell-space path integral for independent verification of gravitational Refractive Depth.
-   Cosmological tests with next-generation observatories (e.g., LISA, SKA).
-   Practical applications of GyroSI and GyroDiagnostics.