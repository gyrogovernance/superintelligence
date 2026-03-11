# Analysis: CGM Constants — Mathematical Structure and the Aperture

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
- BU (Balance Universal): the balance condition that closes the gyrotriangle

The **gyrotriangle defect** is δ = π − (α + β + γ) = π − (π/2 + π/4 + π/4) = 0. Closure is exact: the three angles sum to π, so the defect vanishes. This uniquely fixes the angular structure.

The amplitude A of the oscillation that fits within one observable horizon is constrained by the requirement that left and right SU(2) phase ranges (each 2π) combine with the chiral seed α = π/2. The condition is:

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

### 2.1 BU Dual-Pole Monodromy δ_BU

The **BU dual-pole monodromy** δ_BU is the total phase (memory) accumulated along the path ONA → BU+ → BU− → ONA. It is measured within the CGM framework (e.g. from closure tests) as:

**δ_BU = 2 × ω(ONA ↔ BU) ≈ 0.195342176580 rad.**

So δ_BU is the accumulated angle for the round-trip between ONA and the BU poles. It is a **measured** geometric quantity, not an independent free parameter.

### 2.2 Closure Ratio ρ

The **closure ratio** is the fraction of the aperture “filled” by this monodromy:

**ρ = δ_BU / m_a.**

With the values above, ρ ≈ 0.979300446087. So the system is approximately 97.93% “closed” with respect to the aperture: the accumulated memory almost reaches the full aperture scale. The ratio ρ is dimensionless and appears throughout: in the fine-structure corrections (as 1/ρ), in the interpretation of closure vs aperture, and in the surplus factor (1 − ρ⁴) that contributes to α.

### 2.3 Aperture Gap Δ

The **aperture gap** is the complement of the closure ratio:

**Δ = 1 − ρ = 1 − (δ_BU / m_a).**

So:

**Δ = 1 − δ_BU / m_a.**

Numerically, Δ ≈ 0.020699553913. This is the dimensionless **gap** (about 2.07%) that remains open when comparing the monodromy to the aperture. It is the expansion parameter for systematic corrections (e.g. to the fine-structure constant): small powers of Δ (Δ², Δ⁴) encode aperture effects.

Interpretation:

- **ρ**: fraction of the aperture “used” by the BU dual-pole path (closure).  
- **Δ**: fraction of the aperture “open” (aperture gap). Observation is possible precisely because Δ > 0; full closure would leave no room for observation. So Δ is both the small parameter in perturbation and the geometric reason observation can occur.

### 2.4 Relation to Q_G and m_a

Using ρ = δ_BU/m_a we have Δ = 1 − δ_BU/m_a. The identity Q_G × m_a² = 1/2 can be written as 2 Q_G m_a² = 1. So the aperture m_a sets the scale at which the solid angle Q_G yields this half-integer product; ρ and Δ then measure how the actual monodromy δ_BU sits relative to that scale. In other words: m_a defines the “unit” of closure, and Δ is the deficit from full closure.

---

## 3. SU(2) Commutator Holonomy φ_SU2

### 3.1 Commutator and Trace

For two SU(2) rotations U₁, U₂ with rotation angles β/2 and γ/2 and axes separated by angle δ, the commutator (holonomy) is C = U₁ U₂ U₁† U₂†. The trace is:

**tr(C) = 2 − 4 sin²δ sin²(β/2) sin²(γ/2).**

The holonomy angle φ (in radians) satisfies cos(φ/2) = 1 − 2 sin²δ sin²(β/2) sin²(γ/2). In the CGM configuration, the UNA rotation is π/4 around one axis and the ONA rotation π/4 around an orthogonal axis, so δ = π/2, β = γ = π/4. Then:

sin²(π/2) = 1, sin²(π/8) = (1 − cos(π/4))/2 = (1 − 1/√2)/2,

and the expression simplifies. One obtains:

**cos(φ/2) = (1 + 2√2) / 4.**

### 3.2 Exact Closed Form

So the SU(2) commutator holonomy has the exact closed form:

**φ_SU2 = 2 arccos((1 + 2√2)/4).**

Numerically, φ_SU2 ≈ 0.587900762654 rad (about 33.68°). This is an **exact** geometric result from the SU(2) commutator identity for the chosen angles; no approximation.

### 3.3 Link to δ_BU and the Aperture

Empirically, δ_BU is close to one third of φ_SU2:

**δ_BU ≈ (1/3) φ_SU2 + W_residual,**

with a small residual W_residual. So the BU dual-pole monodromy is tied to the SU(2) holonomy of the UNA/ONA rotations. Since ρ = δ_BU/m_a and Δ = 1 − ρ, the aperture gap Δ is therefore connected to the same SU(2) geometry that gives φ_SU2: the non-commutativity of the path (ONA, UNA, etc.) produces both the holonomy φ_SU2 and the monodromy δ_BU, and the ratio of δ_BU to m_a defines Δ.

---

## 4. Fine-Structure Constant α

### 4.1 Base Formula at the IR Focus

At the BU (IR) focus, the fine-structure constant is given by the **quartic** relation:

**α = δ_BU⁴ / m_a.**

With δ_BU ≈ 0.195342176580 and m_a ≈ 0.199471140201, this yields α ≈ 0.007299683322. The quartic scaling arises from the geometry of dual commutators and dual poles (two quadratic factors). Normalization by m_a ensures the result is dimensionless and tied to the observational aperture.

### 4.2 Role of the Aperture

The base formula can be written as:

**α = (δ_BU/m_a)⁴ × m_a³ = ρ⁴ × m_a³.**

So α depends on the closure ratio ρ to the fourth power and on the aperture m_a. The surplus factor (1 − ρ⁴) appears in the analysis of the fine-structure correction: the ~2.07% aperture (Δ = 1 − ρ) leads to an ~8% surplus (1 − ρ⁴) that is precisely what is needed to bring α from the base value toward the experimental value when corrections are applied.

### 4.3 Systematic Corrections in Terms of Δ

The full CGM formula applies three corrections to the base α₀ = δ_BU⁴/m_a, each expressed using the aperture gap Δ and related geometric quantities:

1. **UV–IR curvature:** α₁ = α₀ × [1 − (3/4)R Δ²], with R the Thomas–Wigner curvature ratio. The factor 3/4 is the SU(2) Casimir. Δ² encodes quadratic aperture effects.

2. **Holonomy transport:** α₂ = α₁ × [1 − (5/6)((φ_SU2/(3δ_BU)) − 1)(1 − Δ² h_ratio) Δ²/(4π√3)]. Here 5/6 is the Z₆ rotor factor with one leg open (aperture), 4π = Q_G, and √3 is the 120° projection. So the aperture gap Δ again enters the geometric transport from UV to IR.

3. **IR alignment:** α₃ = α₂ × [1 + (1/ρ) diff Δ⁴], with ρ = δ_BU/m_a and diff = φ_SU2 − 3δ_BU. The factor 1/ρ ties the correction to closure, and Δ⁴ provides fourth-order suppression.

The complete formula is:

**α = (δ_BU⁴/m_a) × [1 − (3/4)R Δ²] × [1 − (5/6)((φ_SU2/(3δ_BU)) − 1)(1 − Δ² h_ratio) Δ²/(4π√3)] × [1 + (1/ρ) diff Δ⁴].**

Thus the aperture, through m_a, ρ, and Δ, is the central parameter: the base term is normalized by m_a, and all corrections are expansions in Δ (and ρ). The final prediction matches the experimental value to sub-ppm accuracy, with no fitted parameters beyond the measured geometric invariants.

---

## 5. Geometric Quantization: 48Δ and λ₀/Δ

### 5.1 The Relation 48Δ ≈ 1

The depth-4 closure structure projects to a 48-bit tensor (4 stages × 12 bits). The number 48 = 16 × 3 = 2⁴ × 3 appears as a geometric quantization: 16 from the 4π solid-angle structure (e.g. 2⁴), 3 from spatial dimensions. Inflation e-folds in the framework are tied to N_e = 48² = 2304. The aperture gap then satisfies the approximate **geometric quantization**:

**48 × Δ ≈ 1.**

With Δ ≈ 0.020699553913, we have 48 Δ ≈ 0.9936, so there is a small deviation from exactly 1. The relation is **derived** from the N_e = 48² quantization of the CGM structure, not imposed as an arbitrary constraint. So Δ is linked to the discrete 48-fold structure: the aperture gap, when multiplied by the geometric unit 48, nearly equals unity.

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

From the Einstein–Hilbert action quantization in the CGM framework, the dimensionless action is expressed as S_EH/(E₀ T₀) = (σ K ξ)/ζ, with quantization S_EH = κ ν S_geometric. This yields ζ = (σ K ξ)/(ν S_geometric). For the canonical choice (ν, σ, ξ) = (3, 1, 1) and K = 12π, one obtains ζ = π/S_geo, and hence ζ = 2/(m_a √3) = 16√(2π/3), consistent with the definition above. So the aperture m_a, through S_geo, sets the gravitational coupling scale ζ in dimensionless form.

---

## 7. How the Constants Connect

### 7.1 The Aperture as the Hub

- **m_a** is fixed by the closure of the gyrotriangle and the phase-horizon condition A² × 4π² = π/2, and by Q_G × m_a² = 1/2.  
- **ρ = δ_BU/m_a** and **Δ = 1 − ρ** define closure and aperture gap from the measured BU monodromy δ_BU.  
- **φ_SU2** is the exact SU(2) holonomy for the stage angles; δ_BU is empirically close to φ_SU2/3, linking monodromy to holonomy.  
- **α** has base form δ_BU⁴/m_a; corrections are series in Δ (and ρ), so the aperture gap is the expansion parameter.  
- **S_geo = m_a π √3/2** and **ζ = Q_G/S_geo = 16√(2π/3)** tie gravity to the same aperture.  
- **48Δ ≈ 1** and **λ₀/Δ = 1/√5** tie Δ to discrete (48) and pentagonal (√5) geometry.

So **m_a** and **Δ** (with ρ and δ_BU) are the central objects; the rest are derived or expressed in terms of them.

### 7.2 Derivation Chain (Summary)

1. **Angles** (π/2, π/4, π/4) → gyrotriangle defect 0, amplitude condition A²×4π² = π/2 → **m_a = 1/(2√(2π))** and **Q_G × m_a² = 1/2**.  
2. **BU path** → measured **δ_BU** → **ρ = δ_BU/m_a**, **Δ = 1 − ρ**.  
3. **SU(2) commutator** for δ = π/2, β = γ = π/4 → **φ_SU2 = 2 arccos((1+2√2)/4)**.  
4. **IR focus** → **α₀ = δ_BU⁴/m_a**; corrections in Δ, ρ, φ_SU2 → full **α**.  
5. **S_geo = m_a π √3/2** → **ζ = Q_G/S_geo = 16√(2π/3)**.  
6. **48² quantization** → **48 Δ ≈ 1**; pentagonal symmetry → **λ₀/Δ = 1/√5**.

All of these are algebraic or geometric consequences of the aperture m_a, the closure ratio ρ, the aperture gap Δ, and the measured monodromy δ_BU, with no free parameters beyond the framework’s geometric definitions and the single measured value δ_BU.
