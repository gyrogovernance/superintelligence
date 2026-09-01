# Analysis: CGM Constants: Mathematical Structure and the Aperture

This document is an extensive mathematical analysis of the fundamental constants of the Common Governance Model (CGM), with particular emphasis on the aperture parameter and how all quantities derive from and connect to it. Notation uses Unicode symbols throughout; no LaTeX.

---

## 1. The Observational Aperture m_a

### 1.1 Definition and Origin

The **observational aperture** (or amplitude bound) is defined as:

**m_a = 1 / (2√(2π)).**

Numerically, m_a ≈ 0.199471140201. It is an exact closed-form constant, not a fitted value.

Within CGM, observation is treated as a self-referential process: spacetime observes itself into coherence. Coherent observation is bounded by a phase horizon of π radians (half of the full 2π phase structure). The four-stage structure (CS, UNA, ONA, BU) assigns to each stage a threshold angle:

- CS (Common Source): α = s_p = π/2  
- UNA (Unity Non-Absolute): β such that u_p = cos(π/4) = 1/√2  
- ONA (Opposition Non-Absolute): γ = o_p = π/4  
- BU (Balance Universal): the balance condition that closes the gyrotriangle algebraically

The **gyrotriangle defect** is δ = π − (α + β + γ) = π − (π/2 + π/4 + π/4) = 0. Algebraic closure is exact: the three angles sum to π, so the defect vanishes. This uniquely fixes the angular structure. At BU the six kinematic degrees of freedom (three rotational from UNA, three translational from ONA) are coordinated at depth-four commutative closure. A separate **vibrational** degree of motion remains: bounded back-and-forth oscillation about that closed configuration, not a seventh kinematic DOF.

The amplitude A of that oscillation that fits within one observable horizon is constrained by the left and right SU(2) phase ranges associated with the two chiral copies in the double cover of the Lorentz group. Writing (2π)_L and (2π)_R for those full phase ranges and α = π/2 for the chiral seed, the amplitude condition is

**A² × (2π)_L × (2π)_R = α**

i.e. A² × 4π² = π/2. Solving:

**A² = (π/2) / (4π²) = 1/(8π),**

so **A = 1/√(8π) = 1/(2√(2π)) = m_a.**

Thus m_a is the maximum amplitude that keeps the system within the π-radian observable horizon; larger amplitudes would exceed this horizon and accumulate defect. In this sense m_a is the **aperture** through which observation is possible: it both bounds and enables coherent traversal.

### 1.2 Quantum Gravity Invariant and the Aperture

The framework identifies the quantum gravity invariant with the complete solid angle in three dimensions:

**Q_G = 4π.**

This is interpreted as the solid angle required for coherent observation, not as a coupling constant. The horizon length in the construction is L = √(2π). The aperture enters as the time-like (or scale) parameter t_aperture = m_a. The ratio:

**Q_G = L / t_aperture = √(2π) / m_a**

reproduces 4π when m_a = 1/(2√(2π)), since √(2π) × 2√(2π) = 4π. So the aperture is the scale that makes the horizon-to-aperture ratio equal to the full solid angle.

A central identity is:

**Q_G × m_a² = 1/2.**

With Q_G = 4π and m_a = 1/(2√(2π)) we have:

4π × 1/(8π) = 1/2.

This identity links the observational solid angle Q_G to the aperture: the product of the full solid angle with the square of the aperture is exactly 1/2. The half-integer connects to SU(2) double-cover structure (spin-1/2). So m_a is not arbitrary: it is fixed by the requirement that observation be coherent and that Q_G m_a² take this half-integer value.

### 1.3 Geometric Mean Action S_geo

From the aperture we define a **geometric mean action**:

**S_geo = m_a × π × (√3/2) = m_a π √3 / 2.**

With m_a ≈ 0.199471, S_geo ≈ 0.542700940919. This quantity has the dimension of action (angle × scale) and appears in the normalization of gravitational coupling (zeta factor) and in the construction of dimensionless stage actions. The factor √3/2 is the altitude of the equilateral triangle (or 120° rotor projection), tying the aperture to the same triangular/pentagonal geometry that appears in λ₀/Δ and in the SU(2) holonomy.

### 1.4 Chirality and Optical Conjugacy

The CS threshold s_p = π/2 satisfies:

**s_p / m_a² = 4π².**

Since m_a² = 1/(8π), we have (π/2) × 8π = 4π². So the primordial chirality angle, when normalized by the squared aperture, gives the factor 4π² that appears in the optical conjugacy relation E^UV × E^IR = (E_CS × E_EW)/(4π²). The aperture thus links chirality at the source to the geometric dilution between UV and IR foci.

---

## 2. Closure Ratio ρ and Aperture Gap Δ

### 2.1 BU Dual-Pole Loop Angle δ_BU

The **BU dual-pole loop angle** δ_BU is the total phase (memory) accumulated along the dual-pole path. The path departs from the depth-two boundary of the nested lemmas, crosses Balance Egress (BU+) and Balance Ingress (BU−), and returns. In stage coordinates that boundary sits at the common lemma angle π/4 = θ_UNA = θ_ONA, so the path may be written ONA → BU+ → BU− → ONA. Each CGM threshold number is read as an Einstein speed β = ||v|| in the open unit ball, and the Poincaré half-rapidity radius is k(β) = β / (1 + √(1 − β²)) = tanh(atanh(β)/2). The closed form is

**δ_BU = 4 · arctan( k(π/4) · k(m_a) ).**

This equation is the definition of δ_BU. The derivation, the nesting of the stage angles, and the finite realization of the dual poles are given in the Holonomy analysis. So δ_BU is a derived geometric quantity, not an independent free parameter.

### 2.2 Closure Ratio ρ

The **closure ratio** is the fraction of the aperture filled by this loop angle:

**ρ = δ_BU / m_a.**

With the values above, ρ ≈ 0.979300454497. So the system is approximately 97.93% closed with respect to the aperture: the accumulated memory almost reaches the full aperture scale. The ratio ρ is dimensionless and appears throughout: in the fine-structure corrections (as 1/ρ), in the interpretation of closure vs aperture, and in the surplus factor (1 − ρ⁴) that contributes to α.

### 2.3 Aperture Gap Δ

The **aperture gap** is the complement of the closure ratio:

**Δ = 1 − ρ = 1 − (δ_BU / m_a).**

So:

**Δ = 1 − δ_BU / m_a.**

Numerically, Δ ≈ 0.020699545503. This is the dimensionless **gap** (about 2.07%) that remains open when comparing the loop angle to the aperture. It is the expansion parameter for systematic corrections (e.g. to the fine-structure constant): small powers of Δ (Δ², Δ⁴) encode aperture effects.

Interpretation:

- **ρ**: fraction of the aperture scale used by the BU dual-pole loop angle (structural closure in phase).
- **Δ**: fractional **vibrational** amplitude remaining open (about 2.07%). Observation is possible precisely because Δ > 0; if the loop angle saturated the full aperture scale with no residual oscillation, there would be no room for observation. So Δ is both the perturbation expansion parameter and the geometric measure of vibrational motion at BU.

### 2.4 Relation to Q_G and m_a

Using ρ = δ_BU/m_a we have Δ = 1 − δ_BU/m_a. The identity Q_G × m_a² = 1/2 can be written as 2 Q_G m_a² = 1. So the aperture m_a sets the scale at which the solid angle Q_G yields this half-integer product; ρ and Δ then measure how δ_BU sits relative to that scale. In other words: m_a defines the unit of closure, and Δ is the deficit from full closure.

---

## 3. SU(2) Commutator Holonomy φ_SU2

### 3.1 Commutator and Trace

For two SU(2) rotations U₁, U₂ with rotation angles β/2 and γ/2 and axes separated by angle δ, the commutator (holonomy) is C = U₁ U₂ U₁† U₂†. The trace is:

**tr(C) = 2 − 4 sin²δ sin²(β/2) sin²(γ/2).**

The holonomy angle φ (in radians) satisfies cos(φ/2) = 1 − 2 sin²δ sin²(β/2) sin²(γ/2). In the CGM configuration, the UNA rotation is π/4 around one axis and the ONA rotation π/4 around an orthogonal axis, so δ = π/2, β = γ = π/4. Then sin²(π/2) = 1 and sin²(π/8) = (1 − 1/√2)/2, and

```
cos(φ/2) = 1 − 2 · 1 · ((1 − 1/√2)/2)²
         = 1 − (1 − 1/√2)² / 2
         = 1 − (1 − 2/√2 + 1/2) / 2
         = 1 − (3/2 − √2) / 2
         = 1 − 3/4 + √2/2
         = 1/4 + √2/2
         = (1 + 2√2) / 4.
```

### 3.2 Exact Closed Form

So the SU(2) commutator holonomy has the exact closed form:

**φ_SU2 = 2 arccos((1 + 2√2)/4).**

Numerically, φ_SU2 ≈ 0.587900762654 rad (about 33.68°). This is an **exact** geometric result from the SU(2) commutator identity for the chosen angles; no approximation.

### 3.3 Link to δ_BU and the Aperture

The quantities φ_SU2 and δ_BU are distinct constructions: φ_SU2 is the conjugacy angle of the UNA/ONA SU(2) commutator, while δ_BU is the SO(3) dual-pole loop angle under stage speeds as Einstein betas. Their numerical comparison at the canonical thresholds is

```
φ_SU2 / 3 = 0.1959669208846734
δ_BU      = 0.1953421782576621
diff      = φ_SU2 − 3 δ_BU = 0.0018742278810340
```

so δ_BU differs from φ_SU2/3 by about 0.319 percent of φ_SU2. Writing W_residual = δ_BU − φ_SU2/3 ≈ −0.00062474 gives the observed approximate proportion

**δ_BU = φ_SU2 / 3 + W_residual.**

The fine-structure corrections of Section 4 use the defined residual diff = φ_SU2 − 3 δ_BU as an expansion slot. Since ρ = δ_BU/m_a and Δ = 1 − ρ, the aperture gap is built from δ_BU and m_a, while φ_SU2 enters the correction chain through that residual.

---

## 4. Fine-Structure Constant α

### 4.1 Base Formula at the IR Focus

At the BU (IR) focus, the fine-structure constant is given by the **quartic** relation:

**α = δ_BU⁴ / m_a.**

With δ_BU from the closed form of Section 2.1 and m_a ≈ 0.199471140201, this yields α₀ ≈ 0.007299683573. The quartic scaling arises from the geometry of dual commutators and dual poles (two quadratic factors). Normalization by m_a ensures the result is dimensionless and tied to the observational aperture.

### 4.2 Role of the Aperture

The base formula can be written as:

**α = (δ_BU/m_a)⁴ × m_a³ = ρ⁴ × m_a³.**

So α depends on the closure ratio ρ to the fourth power and on the aperture m_a. The surplus factor (1 − ρ⁴) appears in the analysis of the fine-structure correction: the ~2.07% aperture (Δ = 1 − ρ) leads to an ~8.03% surplus (1 − ρ⁴ ≈ 0.08026) that enters the correction chain bringing α₀ toward the experimental value.

### 4.3 Systematic Corrections in Terms of Δ

The full CGM formula applies three sequential refinements α₀ → α₁ → α₂ → α₃ to the base α₀ = δ_BU⁴/m_a, each expressed using the aperture gap Δ and related geometric quantities:

1. **UV–IR curvature:** α₁ = α₀ × [1 − (3/4)R Δ²], with R the Thomas–Wigner curvature ratio. The factor 3/4 is the SU(2) Casimir. Δ² encodes quadratic aperture effects.

2. **Commutator transport:** α₂ = α₁ × [1 − (5/6)((φ_SU2/(3δ_BU)) − 1) Δ²/(4π√3)]. Here 5/6 is the Z₆ rotor factor with one leg open (aperture), 4π = Q_G, and √3 is the 120° projection. So the aperture gap Δ again enters the geometric transport from UV to IR.

3. **IR alignment:** α₃ = α₂ × [1 + (1/ρ) diff Δ⁴], with ρ = δ_BU/m_a and diff = φ_SU2 − 3δ_BU. The factor 1/ρ ties the correction to closure, and Δ⁴ provides fourth-order suppression.

The complete formula is:

**α = (δ_BU⁴/m_a) × [1 − (3/4)R Δ²] × [1 − (5/6)((φ_SU2/(3δ_BU)) − 1) Δ²/(4π√3)] × [1 + (1/ρ) diff Δ⁴],**

with R = 0.993434896272 (Thomas–Wigner curvature ratio) and diff = φ_SU2 − 3δ_BU. Evaluating with δ_BU from Section 2.1 gives α ≈ 0.007297352815. Relative to CODATA 2018 (α = 1/137.035999084), the base α₀ differs by about 319.43 ppm and the fully corrected value by about 33.7 ppb. Thus the aperture, through m_a, ρ, and Δ, is the central parameter: the base term is normalized by m_a, and all corrections are expansions in Δ (and ρ).

---

## 5. Geometric Quantization: 48Δ and λ₀/Δ

### 5.1 The Relation 48Δ ≈ 1

The depth-4 closure structure projects to a 48-bit tensor (4 stages × 12 bits). The number 48 = 16 × 3 = 2⁴ × 3 appears as a geometric quantization: 16 from the 4π solid-angle structure (e.g. 2⁴), 3 from spatial dimensions. Inflation e-folds in the framework are tied to N_e = 48² = 2304. The aperture gap then satisfies the approximate **geometric quantization**:

**48 × Δ ≈ 1.**

With Δ ≈ 0.020699545503, we have 48 Δ ≈ 0.993578, so there is a small deviation from exactly 1. The relation is **derived** from the N_e = 48² quantization of the CGM structure, not imposed as an arbitrary constraint. So Δ is linked to the discrete 48-fold structure: the aperture gap, when multiplied by the geometric unit 48, nearly equals unity.

### 5.2 Pentagonal Symmetry: λ₀/Δ = 1/√5

The **pentagonal** (golden-ratio) constant √5 appears in the CGM geometry. The scale λ₀ is related to the aperture gap by:

**λ₀ / Δ = 1/√5.**

So λ₀ = Δ/√5. Numerically, 1/√5 ≈ 0.447213595500. This is a **derived** geometric relationship from the pentagonal symmetry of the framework, not a separate free parameter. It ties the aperture gap Δ to the golden-ratio geometry (√5) and thus to the same structural family as the 120° rotor and the √3 factors elsewhere.

---

## 6. Zeta Factor ζ (Gravitational Coupling)

### 6.1 Definition from Geometric Invariants

The **zeta factor** is the ratio of the complete solid angle to the geometric mean action:

**ζ = Q_G / S_geo.**

With Q_G = 4π and S_geo = m_a × π × (√3/2):

**ζ = 4π / (m_a π √3/2) = 8 / (m_a √3).**

Substituting m_a = 1/(2√(2π)):

**ζ = 8 × 2√(2π) / √3 = 16√(2π) / √3 = 16√(2π/3).**

Numerically, ζ ≈ 23.155240145865. So ζ is **exactly** determined by the aperture m_a and the geometric action S_geo: it is the ratio of the full solid angle to the action scale set by m_a and the √3/2 factor.

### 6.2 Einstein–Hilbert Connection

From the Einstein–Hilbert action quantization in the CGM framework, the dimensionless action is expressed as S_EH/(E₀ T₀) = (σ K ξ)/ζ, with quantization S_EH = κ ν S_geometric. This yields ζ = (σ K ξ)/(ν S_geometric). For the canonical choice (ν, σ, ξ) = (3, 1, 1) and K = 12π, one obtains **ζ = (12π)/(3 S_geometric) = 4π/S_geometric**. Identifying S_geometric with **S_geo = m_a π √3/2** gives **ζ = Q_G/S_geo** with **Q_G = 4π**, i.e. **ζ = 8/(m_a √3) = 16√(2π/3)**. Therefore the aperture m_a, through S_geo, sets the gravitational coupling scale ζ in dimensionless form.

---

## 7. How the Constants Connect

### 7.1 The Aperture as the Hub

- **m_a** is fixed by the closure of the gyrotriangle and the phase-horizon condition A² × 4π² = π/2, and by Q_G × m_a² = 1/2.  
- **ρ = δ_BU/m_a** and **Δ = 1 − ρ** define closure and aperture gap from the closed-form BU loop angle δ_BU.
- **φ_SU2** is the exact SU(2) commutator angle for the stage angles. The observed proportion δ_BU = φ_SU2/3 + W_residual holds with W_residual ≈ −0.00062474; the residual diff = φ_SU2 − 3δ_BU enters the α correction chain.- **α** has base form δ_BU⁴/m_a; corrections are series in Δ (and ρ), so the aperture gap is the expansion parameter.  
- **S_geo = m_a π √3/2** and **ζ = Q_G/S_geo = 16√(2π/3)** tie gravity to the same aperture.  
- **48Δ ≈ 1** and **λ₀/Δ = 1/√5** tie Δ to discrete (48) and pentagonal (√5) geometry.

So **m_a** and **Δ** (with ρ and δ_BU) are the central objects; the rest are derived or expressed in terms of them.

### 7.2 Derivation Chain (Summary)

1. **Angles** (π/2, π/4, π/4) → gyrotriangle defect 0, amplitude condition A²×4π² = π/2 → **m_a = 1/(2√(2π))** and **Q_G × m_a² = 1/2**.  
2. **BU path** → closed-form **δ_BU = 2ω** → **ρ = δ_BU/m_a**, **Δ = 1 − ρ**.  
3. **SU(2) commutator** for δ = π/2, β = γ = π/4 → **φ_SU2 = 2 arccos((1+2√2)/4)**.  
4. **IR focus** → **α₀ = δ_BU⁴/m_a**; corrections in Δ, ρ, φ_SU2 → full **α**.  
5. **S_geo = m_a π √3/2** → **ζ = Q_G/S_geo = 16√(2π/3)**.  
6. **48² quantization** → **48 Δ ≈ 1**; pentagonal symmetry → **λ₀/Δ = 1/√5**.

All of these are algebraic or geometric consequences of the aperture m_a, the closure ratio ρ, the aperture gap Δ, and the closed-form loop angle δ_BU, with no free parameters beyond the framework’s geometric definitions.
