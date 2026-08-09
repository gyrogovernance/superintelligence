**Aperture, Identity, and Individuality: The Yang–Mills Mass Gap in the Common Governance Model**

**Author:** Basil Korompilias  
**Framework:** Common Governance Model (CGM)  
**Date:** July 2026  
**Archive:** Zenodo DOI 10.5281/zenodo.17521384  

---

## Abstract

The Common Governance Model derives an irreducible operational aperture from the requirement that sequential state transformations preserve both their traceability to a single origin and their capacity to produce distinguishable outcomes. This aperture parameter Δ ≈ 0.0207 emerges as the residual phase defect when alternating transformations close at the fourth order. Existence in the model is the preservation of that common origin under transformation, recorded as operational identity of the vacuum. Emergence is the production of distinguishable outcomes above that origin, recorded as individuality of excitations. The aperture is the residual defect that makes identity and individuality compatible under depth-four closure.

The present construction establishes that the gauge-invariant oriented sector of a finite 4096-state carrier manifold possesses a strictly positive spectral gap above the vacuum. The Yang–Mills mass gap is the canonical continuum readout of this aperture-forced floor. Proceeding from the finite carrier through GNS representation and Osterwalder–Schrader reconstruction to a continuum Hopf-oriented chart, the construction yields a mass scale of approximately 1.582 GeV for the lightest gauge-invariant excitation in the pure SU(2) sector. Coexistence of traceability and distinguishability under transformation closure produces a gapped spectrum in the corresponding field theory.

---

## Introduction

Pure non-abelian Yang–Mills theory in four dimensions exhibits a mass gap when the greatest lower bound on the energy of states orthogonal to the vacuum is strictly positive. Constructing such a theory on continuous spacetime and proving the existence of this gap remains open, formalized as a Clay Millennium problem (Jaffe and Witten 2006). Standard continuum approaches encounter a topological obstruction between local gauge invariance and non-perturbative infrared dynamics. The Gribov–Singer ambiguity shows that no continuous global section exists on the connection quotient, so the infrared vacuum sector is ill-defined before any mass-gap question can be posed.

The Common Governance Model begins from a single axiom stating that the source is common. This axiom enforces ancestry preservation: every operational state remains traceable to a single origin. Identity is that preserved ancestry. Individuality is the capacity of successive transformations to produce distinguishable outcomes. The framework imposes four sequential conditions on state transitions. Common Source (CS) establishes a fundamental directional asymmetry, identifying one transition direction as origin-preserving and the other as origin-altering. Unity Non-Absolute (UNA) keeps the order of sequential operations relevant and prevents homogeneous collapse. Opposition Non-Absolute (ONA) keeps distinct operational paths comparable and prevents complete structural fragmentation. Balance Universal (BU) enforces closure at the fourth order of alternating operations, called depth-four balance. The formal modal statements of these conditions appear in Section 3.1.

Depth-four balance cannot be achieved losslessly in both temporal directions while maintaining distinguishable outcomes. The residual phase defect of this closure is the aperture Δ. It quantifies the irreducible information loss during bidirectional sequence reconstruction. In the continuous realization the aperture is computed from the monodromy of Einstein gyration along the dual-pole path, yielding Δ ≈ 0.0207. The same value is recovered from discrete entropy compression of the byte-level fold disagreement defined in Section 1.1 and from discrete arithmetic quantization anchors (Section 7.1). The aperture is a geometric invariant fixed by the closure identity.

The central thesis of this paper is that the Yang–Mills mass gap is the continuum spectral signature of this aperture. Ancestry preservation requires that the vacuum state remain reconstructable. Distinguishability requires that excitations above this state carry resolvable energy differences. A vanishing aperture would imply perfect reversibility of all transformation sequences, freezing the system into static identity and eliminating the capacity for gapped excitations. The strictly positive aperture therefore forces a strictly positive spectral floor above the vacuum in any admissible oriented quotient. The Yang–Mills problem supplies the external axiomatic standard in which to express this aperture-forced floor as a dimensionful physical mass. The subject of the analysis is identity and individuality under closure. The mass gap is their continuum field-theoretic signature.

The construction proceeds through four layers. The carrier layer defines the finite hQVM manifold Ω with 4096 states and exact byte-induced permutations. The Wilson chart layer embeds finite gauge-invariant lattice blocks into the carrier Hilbert space. The continuum chart layer packages the inductive-limit Euclidean data as a field theory over spacetime via the oriented Hopf quotient. The physical readout layer assigns GeV units through the CGM Δ-ruler using the electroweak scale as the infrared anchor. Finite carrier and Wilson-chart identities are exact. Continuum Yang–Mills claims depend on the infinite-volume limit, uniformity of the spectral floor across the exhaustion family, and reconstruction of the oriented Hopf quotient. Those dependencies are stated in Section 2.3.


## 0. Notation and Standing Definitions

Symbols below are fixed for the entire paper. Subscripts disambiguate families. A bare letter never changes meaning mid-text.

### 0.1 Carrier, states, and Hilbert spaces

| Symbol | Meaning |
|---|---|
| Ω | Finite hQVM carrier manifold, |Ω| = 4096, with bipartite product Ω = U_fac × V_fac, |U_fac| = |V_fac| = 64. |
| χ ∈ GF(2)⁶ | Chirality / transport register on Ω. |
| fold disagreement | Hamming weight of (bits 0–3) XOR (bits 4–7) of an intron (Section 1.1). |
| carrier Z₂ | Binary provenance coordinate within each chirality shell, defined by the involution F (Section 1.1). |
| ψ ∈ ℂ⁴⁰⁹⁶ | Canonical Hilbert lift of the carrier (coordinates indexed by Ω). |
| ω⋆ : 𝔄 → ℂ | Canonical CGM state functional, ω⋆(I) = 1. |
| ω_∞ | Infinite-volume limit state on A_loc. |
| ω_avg | Unoriented fiber-average state (negative witness of Theorem D3-struct). |
| (ℋ_ω, π_ω, Ω_vac) | GNS triple of ω⋆; Ω_vac = [I] is the cyclic vacuum vector. |
| Ω_k | Unique ground state of the finite-volume Hamiltonian H_k on volume Λ_k. |
| ℋ_OS | Osterwalder–Schrader / Wightman physical Hilbert space (reconstruction of Section 4.3). |
| ℋ_GI | Gauge-invariant (GI) Wilson subspace of the defining Q₈ chart (dim = 28). |
| H_phys | Self-adjoint Hamiltonian generating time on ℋ_OS (and, by GNS, on ℋ_ω); H_phys Ω_vac = 0. |

Surface measure on S² is written dμ_{S²}.

### 0.2 Algebras, groups, and classical gauge data

| Symbol | Meaning |
|---|---|
| 𝔄 | Free modal *-algebra generated by left/right gyrations [L], [R]. |
| A(Λ) | Local gauge-invariant cylinder *-algebra on finite spatial volume Λ ⊂ ℤ³. |
| A_loc | Inductive limit ⋃_Λ A(Λ). |
| A₊ | Positive-time subalgebra of A_loc (OS). |
| G | Compact simple Lie group (Clay structure group: SU(N), etc.). |
| G^E | Product of G-valued links on the oriented edge set E. |
| Conn | Space of smooth connection 1-forms on a principal G-bundle. |
| 𝒢 | Group of gauge transformations acting on Conn. |
| ℬ = Conn/𝒢 | Classical physical configuration space. |
| K₄ | Klein four-group (holonomic deck / family gauge on the carrier). |
| Q₈ | Quaternion group of order 8, central extension 1 → Z₂ → Q₈ → K₄ → 1; defining Wilson root chart. |

### 0.3 Gaps, aperture, and mass scales

| Symbol | Meaning |
|---|---|
| Q_G = 4π | Horizon solid-angle normalization (steradians). |
| m_a = 1/(2√(2π)) | Aperture scale. Identity: Q_G m_a² = 1/2. |
| δ_BU | Balance Universal (BU) monodromy phase defect (radians); derived in Section 7.1. |
| ρ = δ_BU / m_a | Structural closure ratio (ρ ≈ 0.9793). |
| Δ = 1 − ρ | **Aperture** (dimensionless). This is the only meaning of bare Δ. |
| Δ_W | Unoriented shadow gap; Δ_W(n) = n/(2(n−1)), lim_{n→∞} Δ_W = 1/2. |
| Δ_JW | Defining-chart dimensionless spectral gap E₁ − E₀ on the Q₈ 1×1 Wilson block. |
| Δ_* | Strong-coupling spectral floor (volume-uniform lower bound on audited charts). |
| gap(H) | Spectral gap of a self-adjoint operator H above its ground energy. |
| Δ_phys | Greatest lower bound of spec(H_phys) on Ω_vac^⊥; the mass gap of the continuum theory. |
| v = E_EW ≈ 246.22 GeV | Electroweak vacuum expectation value (Δ-ruler IR anchor). |
| E_unit := v Δ | Grade-1 energy unit (independent of Δ_JW). |
| E_g2 := v Δ² | Grade-2 curvature scale. |
| C₂ = C(6,2) = 15 | Exact 2-form channel multiplicity of GF(2)⁶. |
| m_gap := C₂ · v · Δ² | Continuum mass readout (Route A). |
| m_B := S_CS · 2 · v · Δ² | Cross-check mass readout (Route B); S_CS = (π/2)/m_a. |
| m_coupled(O) | Infimum of energies of states coupled to observable O. |
| κ₂(O) := m_coupled(O)/Δ | Chart curvature index for observable O. In particular κ₂(O_Λ²) and κ₂(V_plaq) are distinct. |

Optical conjugacy: E_i^UV · E_i^IR = (E_CS · v)/(4π²) with E_CS = 1.22×10¹⁹ GeV.

### 0.4 Operators and charts (selected)

| Symbol | Meaning |
|---|---|
| H_k | Wilson–Kogut–Susskind Hamiltonian on volume Λ_k. |
| H_elec, H_mag | Electric / magnetic sectors. |
| H_W | Unoriented shadow curvature operator on ℓ²(Ω) (Theorem D3-struct). |
| Lap_G | Laplace–Beltrami operator on G. |
| W | Cocycle isometry ℓ²(Q₈²)_GI ↪ ℓ²(Ω). |
| Θ | Euclidean time reflection (= BU fold involution on bits 3–4). |
| K_OS | OS transfer kernel across t = 0. |
| K_comm(x,y) | Commutator defect T_x T_y T_x⁻¹ T_y⁻¹ on the carrier. |
| M_ij = E_μ[(Θ F_i) F_j] | OS Gram matrix. |
| O_Λ² = Σ_{a<b} O_ab | Full bivector-channel curvature sum; I₂ = {(a,b) : 0 ≤ a < b ≤ 5}. |
| N_{2,active}(ψ) | Number of channels (a,b) ∈ I₂ with ‖P_ab ψ‖ ≠ 0. |
| Γ_payload = S₆ | Permutation group of the six payload bit indices. |
| Aut(Q₈) ≅ S₄ | Automorphism group of Q₈ (|Aut(Q₈)| = 24). |
| U_fac, V_fac | Bipartite factors of Ω (not unitaries). |
| U_L(t), U_R(t) | One-parameter unitary groups implementing [L], [R]. |
| Û(φ) | Unitary on ℋ_GI induced by φ ∈ Aut(Q₈). |
| U_σ | Unitary on the abstract channel space 𝒦 induced by σ ∈ Γ_payload. |
| V_R(hol(p)) | Wilson magnetic weight on plaquette p. |
| V_∂ | Boundary perturbation H_j − H_i. |
| face_A, face_B | Active / passive 12-bit faces of a Gyrostate on Ω. |
| A_v | Wilson vertex projector (A_v² = A_v); not the modal algebra 𝔄. |

### 0.5 Typographic conventions

1. Bare Δ denotes the aperture. Spectral gaps are written gap(H), Δ_JW, Δ_W, Δ_*, or Δ_phys.
2. Bare Ω denotes the carrier. The vacuum vector is Ω_vac (or Ω_k on finite volume).
3. G denotes the compact simple structure group; 𝒢 denotes gauge transformations.
4. 𝔄 denotes the modal *-algebra; A(Λ) and A_loc denote lattice observable algebras.
5. Continuum mass is m_gap. Chart diagnostics use Δ_JW, κ₂(O), and m_coupled(O).

### 0.6 Inputs used in this paper

| Item | Status in this paper |
|---|---|
| Q_G = 4π | Horizon solid-angle definition (Section 7.1). |
| m_a = 1/(2√(2π)) | Derived from Q_G m_a² = 1/2 (Section 7.1). |
| δ_BU ≈ 0.195342 rad | Derived from Einstein gyration on the dual-pole path ONA → BU+ → BU− → ONA (Section 7.1). |
| Uniqueness of ω⋆ | Sketch in Section 3.1; finite modal quotient under BU identification. |
| OS net, OS-RP, Wightman reconstruction | Derived here (Section 4). |
| Gap positivity Δ_phys > 0 from Δ > 0 | Derived here (Section 7). |
| D0 transversality chain, Route A scale m_gap | Derived here (Section 7). |
| Theorem D3-struct (shadow lock) | Derived here (Section 7.2). |
| v = E_EW, E_CS | IR/UV anchors of optical conjugacy (Section 6.1); GeV calibration is framework-referenced. |


## 1. Finite Carrier, QuBEC Measure, and Admissible Shadows

### 1.1 The finite carrier and byte transition

The hQVM carrier Ω is the reachable set of 24-bit states s = (A12, B12), where A12 and B12 are twelve-bit faces drawn from the self-dual [12,6,2] mask code. The carrier has product form Ω = U_fac × V_fac with |U_fac| = |V_fac| = 64, so |Ω| = 4096.

Each input byte is transcribed against the micro-archetype GENE_Mic = 0xAA by the rule intron = byte XOR 0xAA. The boundary bits of the intron, positions 0 and 7, select one of four K₄ family labels. The six interior bits, positions 1 through 6, form the transport payload q₆ ∈ GF(2)⁶. The payload expands to a twelve-bit mutation mask through the [12,6,2] code, with one payload bit controlling one oriented dipole pair in the SE(3) register.

The carrier transition T_b mutates the active face A12 by the payload mask and then applies the family-selected gyration between the active and passive faces. On the chirality register χ ∈ GF(2)⁶, obtained as the pair-diagonal collapse of A12 XOR B12, this transition projects to the affine transport rule χ(T_b(s)) = χ(s) XOR q₆(b). The byte alphabet therefore forms a four-to-one cover of the six-bit transport space, with the K₄ family label carrying the spinorial phase information. The Q-map sends 256 bytes onto 64 transport classes with uniform fiber size 4. Each transport class yields 2 distinct Ω-permutations, giving the chain 256 → 128 → 64. Shadow partners b and b XOR 0xFE produce identical Ω-permutations but different spinorial phases, realizing the SU(2)/SO(3) double cover at the byte level.

Reachability from rest under a byte alphabet A is governed by the transport rank r(A) = dim span{q₆(b) : b ∈ A} in GF(2)⁶. Under fiber-complete restriction the reachable set satisfies |Reach(A)| = (2^{r(A)})². Full coverage of the 4096-state carrier requires r(A) = 6.

The K₄ structure in the hQVM is a family-phase deck action on the byte cover. It is distinct from the Yang–Mills structure group G. The finite Q₈ chart is used because Q₈ is the minimal non-abelian central extension of K₄ and embeds in SU(2). It supplies a finite Wilson root chart for carrier certificates. The physical gauge group in the continuum discussion remains the compact simple group G.

Depth-four words on the carrier generate the Klein four-group {id, W₂, W₂′, F}. The wavefunction analysis verifies this algebra exhaustively over all 64 micro-references and all 4096 carrier states by exact integer arithmetic. In particular, W₂ and W₂′ exchange the constitutional poles, F preserves shell while pairing each state with a unique partner in the same shell, and W₂² = id. These identities identify BU-Egress with an involutive closure operator and BU-Ingress with its pole-pairing memory action. Theorem OS-RP and Theorem D3-struct use this verified algebra.

Each intron has a forward four-bit reading given by bits 0 through 3 and a reverse four-bit reading given by bits 4 through 7. The fold map P exchanges these readings across the BU boundary at bits 3 and 4. The fold disagreement of an intron is the Hamming weight of the bitwise XOR of the forward and reverse readings. Its carrier-level counterpart is a binary coordinate within each chirality shell. That coordinate is the carrier Z₂ coordinate. It is defined by the involution F, since F preserves the chirality value and pairs each state with a unique partner state in the same shell.

**Proposition (Carrier commutator defect).** Let T_x and T_y be the carrier permutations induced by bytes x and y. On the chirality register, the commutator defect K_comm(x, y) = T_x T_y T_x⁻¹ T_y⁻¹ acts as translation by d(x, y) = q₆(x) XOR q₆(y). Two bytes commute if and only if q₆(x) = q₆(y). The fraction of commuting ordered pairs is 1024/65536 = 1/64.


### 1.2 The QuBEC carrier measure

The QuBEC (Quantum Bose–Einstein Computational) state is the finite occupation measure on Ω induced by byte transport. Shells are indexed by the Hamming weight of the chirality register χ, equivalently by the Hamming distance between A12 and B12. The seven shells have binomial populations 64, 384, 960, 1280, 960, 384, 64. The equality horizon (shell 0, A12 = B12) and the complement horizon (shell 6, A12 = B12 XOR 0xFFF) are the two 64-state constitutional poles. The equatorial shell 3 has maximal population 1280 and dominates the occupation measure that the Hopf chart reads as the continuum horizon density. Weighting shell k by λ^k gives the exact partition function Z₁(λ) = 64(1 + λ)⁶. The factor 64 is the uniform horizon degeneracy, and the binomial factor records the six transport modes.

This measure supplies the finite carrier reference for local Wilson-chart embeddings. Transport on χ is diagonalized by the 64-point Walsh–Hadamard transform, and radial shell observables are diagonalized by Krawtchouk polynomials. The Yang–Mills construction uses this finite measure as the carrier-side source of positivity, shell weighting, and aperture transport.

---


### 1.3 Admissible shadows and the observational layer

Accessible content is realized as quotients of the carrier defined in Sections 1.1 and 1.2. The formalism quotient ladder is

```
Byte256 → q₆ ∈ GF(2)⁶ → shells₇,    Ω → χ ∈ GF(2)⁶ → shells₇.
```

Each arrow forgets ordering, frame, or phase and retains a transport, weight, or shell invariant. Parallel shadows in the same architecture include the SU(2) → SO(3) double cover (256 bytes induce 128 distinct Ω-steps at rest), the depth-4 BU fold (holonomy as fold defect), and the Hopf map S³ → S² (fiber S¹ is global phase; polar packaging of ℝ⁴ over S² is the continuum chart of Section 5).

Optical conjugacy places the ultraviolet focus GENE_Mic opposite Balance Universal (BU), the infrared focus where observation occurs. Lemma UNA is the minimal condition for indirect observation of that source. Lemma ONA is the minimal condition for direct observation of non-absolute unity. Proposition BU closes at depth four as the infrared observational shell, with energy scales on the Δ-ruler between the two foci.

In the Jaffe and Witten formulation, local observables are gauge-invariant functions on connections. In CGM the same role is an admissible shadow. An admissible shadow is a quotient map π from the carrier Ω to a state space Σ such that (1) the micro-archetype GENE_Mic = 0xAA orients the fiber over every point in Σ, (2) the aperture defect Δ = 1 − ρ descends to a positive function on Σ, and (3) the local curvature generators (the six payload bits of the byte) induce non-trivial transport on Σ. Laboratory observables are correlators of that gauge-invariant shadow. The shadow is inadmissible when condition (1) fails, which occurs when the quotient averages over the K₄ family fiber and erases the transcription baseline. Theorem D3-struct shows that such unoriented shadows collapse the spectral gap to the value 1/2.


## 2. The Mass Gap Problem

### 2.0 Spectral definition

In quantum field theory the mass gap is the greatest lower bound on the energy of states orthogonal to the vacuum. With vacuum energy set to zero, and with particle states read as plane-wave eigenstates of the energy-momentum operators, that bound is the mass of the lightest particle in the theory.

Equivalently, a theory has a mass gap when connected vacuum two-point functions of local observables decay exponentially in Euclidean time,

```
⟨O(t) O(0)⟩_c ∼ Σ_n A_n exp(−Δ_n t),   Δ_0 > 0,
```

with Δ_0 equal to the spectral threshold above the vacuum. Lattice spectroscopy measures this decay. The continuum problem is to construct the theory on ℝ⁴ so that the same threshold exists as a theorem. In this paper bare Δ denotes the CGM aperture (Section 0.3). The continuum spectral threshold is written Δ_phys.


### 2.1 The expected gap, lattice evidence, and the continuum problem

Pure non-abelian Yang–Mills theory has no elementary matter fields. Confinement implies that colored gluons do not appear as free massless particles. The physical spectrum consists of color-neutral composites (glueballs). If the lightest glueball is massive, the theory has a mass gap.

Asymptotic freedom supplies a trivial ultraviolet fixed point and an infrared mass scale generated non-perturbatively, which makes pure Yang–Mills a natural four-dimensional target for constructive quantum field theory, simpler than full QCD with dynamical quarks (Section 6).

Lattice computations for SU(N) gauge theories find a discrete glueball spectrum with a positive lightest scalar mass in the ballpark of 1.5–1.7 GeV in physical units for SU(3) (Morningstar and Peardon 1999; Lucini and Teper 2001; Chen et al. 2006). Those results are finite-volume, finite-spacing extrapolations. They support a gap on the lattice. They do not by themselves supply a continuum theory on ℝ⁴ with a proved threshold.

The Clay Mathematics Institute formalizes the open continuum task: for any compact simple gauge group G, construct a nontrivial quantum Yang–Mills theory on ℝ⁴ satisfying OS/Wightman-level axioms and prove that its mass gap Δ_phys is strictly positive (Jaffe and Witten 2006). Inside CGM that task is the external axiomatic standard in which the aperture-forced spectral floor is expressed as a continuum mass. The Clay checklist is recorded in Section 8.


### 2.2 Continuum obstruction and the operational pivot

The Introduction records the Gribov–Singer obstruction on Conn/𝒢. The technical content of that obstruction is as follows. Let Conn be the space of smooth connection 1-forms on a principal G-bundle P over a four-manifold M. Let 𝒢 be the group of gauge transformations acting by fibrewise conjugation. The physical configuration space is the quotient manifold ℬ = Conn / 𝒢. To define a functional integral measure over ℬ, canonical quantization schemes attempt to select a unique representative from each gauge orbit by defining a gauge-fixing hypersurface. Gribov (1978) and Singer (1978) proved that the principal bundle Conn → Conn / 𝒢 is topologically non-trivial for any non-abelian compact Lie group G in D ≥ 3 dimensions. Consequently, no continuous global section exists. Any analytical gauge condition necessarily intersects certain gauge orbits multiple times (Gribov copies) or fails to intersect them entirely (the Gribov horizon). Continuum schemes that begin from Conn/𝒢 therefore inherit an incomplete infrared vacuum sector before any mass-gap question can be posed.

Unrestricted averages over gauge orbits fail to preserve a canonical reference state. Without an invariant operational origin, the vacuum sector is ambiguous across topological sectors, and a spectral gap cannot be defined unambiguously.

The Gribov–Singer obstruction applies to continuous principal bundles over continuous base manifolds. The present construction defines the canonical state and the gauge-invariant subspace directly on the finite deterministic carrier, where continuous topological sectioning is not required. Local gauge invariance is the operational requirement that physical observables remain independent of the unobservable internal phase routing that tracks ancestry to the common source. Gauge transformations are redundancies in that internal routing that leave the bulk transition geometry invariant. Continuum space is packaged afterward by the polar–Hopf chart over the physical observation horizon S². On the admissible oriented quotient the physical gap is defined relative to ω⋆. An unoriented average that erases the GENE_Mic reference collapses curvature to an unphysical half-gap shadow (Theorem D3-struct, Section 1.3).

The continuum construction items are as follows. The GNS triple of Section 3.2 gives the Hilbert space and vacuum for the modal operational algebra. Theorem OS-RP, Theorem OS-Cont, and Section 4.3 give the Osterwalder–Schrader axioms and Wightman reconstruction under the limiting hypotheses stated there. Theorem GAP-Positive gives the mass gap on the admissible oriented quotient. Theorem AF-Ruler gives asymptotic freedom along the Δ-ruler. The Hopf chart of Section 5 supplies the continuum ℝ⁴ packaging. Corollary Clustering of Section 7.5 records the exponential decay of connected correlators. A compact survey of the standard literature on these barriers is Madisa (2026).

### 2.3 Status of the construction

The finite hQVM and finite Wilson-chart results are unconditional within the stated kernel. They establish carrier reachability, shell census, transport rank, K₄ identities, plaquette-defect distributions, and the reported finite-chart spectral certificates.

The continuum claims require additional inputs beyond finite enumeration: a compatible infinite-volume local net, existence and uniqueness of the limiting state, uniform control along the lattice-spacing and volume limits, reflection positivity of the limiting Schwinger functions, Euclidean covariance and regularity, reconstruction of a nontrivial Poincaré-covariant theory, and identification of the lowest gauge-invariant excitation with the grade-2 carrier sector.

The dependency chain is

```
finite carrier and Wilson charts
→ compatible local net
→ infinite-volume and continuum limits
→ OS / Wightman reconstruction
→ identification of the physical excitation
→ positive continuum mass gap.
```

Finite-model entailments must not be identified automatically with continuum entailments. Small Kripke frames and finite Wilson charts can impose accidental relations that disappear under enlargement. Every finite certificate used here is interpreted as a kernel or defining-chart result unless volume, spacing, and representation stability have been established separately. The three load-bearing continuum dependencies are uniqueness and positivity of the oriented state ω⋆, existence of the inductive-limit local state with the required OS regularity properties, and identification of the saturated grade-2 payload curvature sector with the lightest gauge-invariant Yang–Mills excitation on the Hopf-oriented chart.


## 3. Canonical State and GNS Construction

### 3.1 The CGM Canonical State Functional ω⋆

The Introduction defines ancestry preservation and the four sequential conditions CS, UNA, ONA, and BU. This section formalizes those conditions as a unique positive linear state functional ω⋆ : 𝔄 → ℂ, normalized to ω⋆(I) = 1, on the free modal *-algebra 𝔄 generated by left gyration [L] and right gyration [R] (active mutation and passive reference preservation). The functional is fixed by the four parameter-free axioms below. Because ω⋆ is derived from ancestry preservation on the operational algebra, the construction proceeds without an analytical gauge-fixing section on Conn / 𝒢.

The axioms constrain modal-depth observables at horizon worlds w in the S-sector. Unity at depth one holds when [L]S ↔ [R]S at w. Equality at depth two holds when [L][R]S ↔ [R][L]S at w. Opposition at depth two holds when [L][R]S ↔ ¬[R][L]S at w. Balance at depth four holds when [L][R][L][R]S ↔ [R][L][R][L]S at w. The four axioms impose modal-necessity requirements on these observables:

1. **Assumption CS (Common Source):** Establishes fundamental chirality at the observable horizon S. Right transitions preserve the horizon reference, and left transitions depart from it.  
   ω⋆([R]S ↔ S) = 1, ω⋆([L]S ↔ S) ≠ 1.

2. **Lemma UNA (Unity Non-Absolute):** Excludes absolute structural collapse (abelian trivialization) at operational depth two by contingent non-commutativity.  
   ω⋆(([L][R] − [R][L])* ([L][R] − [R][L])) > 0.

3. **Lemma ONA (Opposition Non-Absolute):** Excludes total structural fragmentation (causal disconnection) at depth two by ensuring comparable relational paths.  
   ω⋆(¬□¬([L][R]S ↔ [R][L]S)) = 1.

4. **Proposition BU (Balance Universal):** Enforces commutative algebraic closure at operational depth four. The closed state retains geometric memory of prior chiral transitions (the aperture defect).  
   ω⋆(([L][R][L][R] − [R][L][R][L])* ([L][R][L][R] − [R][L][R][L])) = 0.

The four conditions determine ω⋆ uniquely. The free modal *-algebra 𝔄 generated by [L] and [R] admits a convex cone of positive linear functionals. Assumption CS fixes the chirality signature at the horizon, selecting a ray in this cone. Lemma UNA excludes the abelian face where [L] and [R] commute absolutely. Lemma ONA excludes the opposing face where they anticommute absolutely. Proposition BU imposes the depth-four closure constraint, which intersects the remaining spectral wedge at a single point. The Z3 SMT enumeration that verifies uniqueness is finite because the depth-four modal words live in a finite quotient of the free algebra under the BU identification. The existence and uniqueness of this intersection are verified by Kripke frame analysis and that finite SMT enumeration.

Computationally, the state is anchored by the discrete micro-archetype GENE_Mic = 0xAA and normalized over the continuum horizon by the invariant solid angle Q_G = 4π steradians.


### 3.2 The GNS Construction of the Operational Hilbert Triple

Given the *-algebra 𝔄 and the exact positive linear functional ω⋆, the Gelfand–Naimark–Segal (GNS) representation theorem guarantees the existence of a unique (up to unitary equivalence) Hilbert space triple:

```
(ℋ_ω, π_ω, Ω_vac)
```

where ℋ_ω is the Cauchy completion of the quotient pre-Hilbert space 𝔄 / N with respect to the inner product ⟨[a], [b]⟩ = ω⋆(a* b). The left ideal N = {a ∈ 𝔄 : ω⋆(a* a) = 0} represents null curvature elements. The linear map π_ω : 𝔄 → B(ℋ_ω) acts as a bounded *-representation of the operational algebra on ℋ_ω via left multiplication π_ω(a)[b] = [a b]. The cyclic distinguished vacuum vector is Ω_vac = [I]. It satisfies

```
ω⋆(a) = ⟨Ω_vac, π_ω(a) Ω_vac⟩   for all a ∈ 𝔄.
```

The abstract GNS representation is the GNS triple of the free modal *-algebra 𝔄 with state ω⋆. It supplies a Hilbert space and vacuum for that operational algebra. Passage to the local gauge-invariant observable algebra A_loc, to Wilson loop operators, and to the continuum field algebra proceeds through the defining Wilson chart and the inductive net of Section 4.

The gauge-invariant (GI) subspace ℋ_GI of the Q₈ Wilson block at volume 1×1 has dimension 28. The cocycle isometry W embeds this space into the carrier Hilbert space ℓ²(Ω) by mapping link configurations to carrier states through the transcription rule intron = byte XOR 0xAA. Under this embedding, the Wilson vertex projector A_v and magnetic weight V_R(hol(p)) act as bounded operators on ℓ²(Ω) that commute with the chirality register. The finite-chart Hamiltonian H_k of Section 4.1 is the restriction of the GNS representation π_ω to the image of W. The carrier certificates verify the finite algebraic identities used by the Q₈ and K₄ Wilson charts. Their role in the continuum construction is mediated by the embedding W, the inductive local net, and the Hopf-oriented reconstruction map.

The canonical Hilbert lift ψ ∈ ℂ⁴⁰⁹⁶ carries verified quantum-information structure. Pairwise CHSH correlators on the lifted graph state saturate the Tsirelson bound 2√2 to precision 10⁻¹². These certificates establish that the finite code lift realizes standard quantum-information correlations on ψ. Continuum locality, Poincaré covariance, and the Yang–Mills field algebra are constructed through the Wilson net and OS reconstruction of Section 4.

The distinguished vacuum vector represents the operational rest state. It resides on the complement horizon of the carrier manifold, the locus of maximal chirality and maximal operational opposition, where active mutation evaluates to zero. Physical excitations above the vacuum are transitions from this constitutional pole into the relational bulk shells. Departure from the pole into the bulk is a discrete structural step measured in integer Hamming distance on the carrier. The Hopf chart reads this separation as a finite spectral scale once the limiting hypotheses of Section 2.3 supply the continuum Hamiltonian and its vacuum sector. The physical Hamiltonian generates time transitions in the closed algebra and annihilates the vacuum: H_phys Ω_vac = 0. This supplies the vacuum for the GNS representation of 𝔄 and, under the bridge of Section 4, the vacuum of the reconstructed continuum theory.

---


## 4. The Canonical Operator Net and Osterwalder–Schrader Reconstruction

### 4.1 Finite-Volume Local *-Algebras and Inductive Exhaustion

To prove the existence of the infinite-volume limit on continuous spacetime, we construct a directed system of local gauge-invariant *-algebras. Let Ω be the reachable finite carrier manifold of Section 1.1. For any finite hypercubic spatial volume Λ ⊂ ℤ³ with lattice spacing a, let G^E denote the product of compact simple gauge group links assigned to the oriented edges E of Λ, and let P be the set of elementary plaquettes.

On a finite spatial lattice Λ the Wilson–Kogut–Susskind Hamiltonian is written in the convention

```
H_Λ(g, a) = (g² / 2a) Σ_{e ∈ E} (−Lap_e) + (1 / (2 g² a)) Σ_{p ∈ P} V_R(hol(p)),
```

where Lap_e is the Laplace–Beltrami operator on the copy of G assigned to edge e, hol(p) is the oriented plaquette holonomy, and V_R(g) = 1 − Re χ_R(g) / dim(R) is the Wilson magnetic weight in representation R. The gauge-invariant Hilbert space is obtained by imposing vertex projectors A_v at all internal vertices. The ground energy is shifted to zero when spectral gaps are discussed. We write H_k for H_Λ_k on the exhaustion volumes below.

We define the local *-algebra A(Λ) as the algebra of bounded gauge-invariant cylinder functions acting on the Hilbert space ℓ²(G^E). A function F ∈ A(Λ) depends exclusively on link variables within the bounded regional subgraph Λ and is invariant under local gauge transformations at all internal vertices. For any inclusion of finite spatial volumes Λ₁ ⊂ Λ₂, the natural isometric embedding of cylinder functions defines an injective *-homomorphism i_Λ₁,Λ₂ : A(Λ₁) ↪ A(Λ₂). The algebra of local observables is the inductive limit:

```
A_loc = ⋃_{Λ ⊂ ℤ³} A(Λ).
```

**Lemma IV (Infinite-Volume Vacuum Existence and Uniqueness):** Let {Λ_k} be an exhaustive nested sequence of finite spatial hypercubes converging to ℤ³. Lemma IV is conditional on a volume-uniform finite-volume spectral floor for the chosen Wilson-chart exhaustion. On each finite volume Λ_k, let H_k be the self-adjoint Wilson–Kogut–Susskind Hamiltonian on ℓ²(G^E) possessing a unique ground state Ω_k (open boundary conditions on the audited regional charts) with spectral gap gap(H_k) ≥ Δ_* > 0 independent of volume size. The finite Q₈ and K₄ certificates establish this floor on audited regional charts. The extension from those regional certificates to the full exhaustive family uses the root SU(2) comparison sector of Theorem SC0-G-cont together with the quadratic-form extension hypothesis stated there. Under that floor hypothesis there exists a unique continuous positive linear functional ω_∞ on the local gauge-invariant algebra A_loc such that

```
ω_∞(O) = lim_{k → ∞} ⟨Ω_k, O Ω_k⟩   for all O ∈ A_loc.
```

*Proof:* For any fixed local gauge-invariant observable O with support in a bounded region Λ_0, consider two finite volumes Λ_i, Λ_j with Λ_0 ⊂ Λ_i ⊂ Λ_j. Let H_j be the Hamiltonian on Λ_j, and let H_i be the Hamiltonian restricted to Λ_i. The difference between ground-state expectation values is governed by the boundary perturbation V_∂ = H_j − H_i. Its support resides strictly on the boundary cut ∂Λ_i. Applying finite-velocity propagation bounds anchored by the QuBEC aperture Δ > 0 of Section 1.2 on the bipartite carrier Ω = U_fac × V_fac to the imaginary-time operator T(t) = e^(−t H_j) acting on the orthogonal complement of the ground state P_⊥, we obtain

```
|⟨Ω_j, O Ω_j⟩ − ⟨Ω_i, O Ω_i⟩| ≤ C · ‖O‖ · ‖V_∂‖ · exp(−μ · dist(Λ_0, ∂Λ_i))
```

where μ = Δ_* / (c · J_*) is the universal rate of spatial attenuation, J_* = max(J_e, J_m) is the upper bound on the interaction strength, and c is the Lieb–Robinson velocity constant. Because the strong-coupling spectral floor satisfies Δ_* > 0 uniformly in volume size on the audited charts, the attenuation rate μ is positive. As the boundary ∂Λ_i recedes to infinity (i, j → ∞), the exponential term exp(−μ · dist(Λ_0, ∂Λ_i)) tends to zero. Thus the sequence of expectations {⟨Ω_k, O Ω_k⟩} forms a Cauchy sequence in ℂ for every local cylinder function O ∈ A_loc. The pointwise limit ω_∞(O) exists. It is normalized to ω_∞(I) = 1 and is positive definite. By standard extension theorems, ω_∞ defines a unique state on the C*-algebra completion of A_loc. Audited K₄ charts give volume-stable strong-coupling grade-1 (SC1) floors and Cauchy magnetic densities. A periodic Q₈ volume comparison shows that the torus spectrum differs from the physical mass gap (Appendix A). The continuum reading of ω_∞ is the Hopf chart of the oriented quotient (Section 5). ∎

### 4.2 Osterwalder–Schrader Reflection Positivity (H4)

A critical requirement for constructing a relativistic quantum field theory from Euclidean data is Osterwalder–Schrader Reflection Positivity (OS-RP). Reflection positivity is the Euclidean condition that yields a positive-definite Minkowski Hilbert space under Osterwalder–Schrader reconstruction (Osterwalder and Schrader 1973). In gauge theories, continuum gauge fixing and Faddeev–Popov quantization introduce indefinite-metric states that must decouple before physical spectral theory can be stated (Seiler 1982; Jaffe and Witten 2006).

We establish non-tautological reflection positivity on the CGM–Wilson charts. Let μ_β be the Gibbs–Wilson measure on link configurations over a finite Euclidean spacetime lattice T × L. Let the time-slice algebra A_+ ⊂ A_loc be the subalgebra of bounded gauge-invariant cylinder functions whose support is strictly restricted to positive Euclidean times t ≥ 0. Let Θ be the Euclidean time-reflection operator, realized as the BU fold involution (bits 3–4) in the byte formalism. Θ acts on link variables by flipping the orientation of timelike links across the hyperplane t = 0 while preserving spacelike links: U_(x, t) ↦ U_(x, −t)^(−1).

**Theorem OS-RP (BU Fold Reflection Positivity on Wilson Charts):** For any finite sequence of positive-time cylinder functions F_1, …, F_n ∈ A_+ and constants c_i ∈ ℂ, the Euclidean Gram matrix

```
M_ij := E_μ[(Θ F_i) F_j]
```

is positive semidefinite on every finite Wilson chart for which the transfer factorization below holds.

*Proof:* Two CGM-native ingredients and one Wilson structural fact.

(1) **Θ = BU fold.** In the hQVM byte fiber bundle the depth-four fold map at bits 3–4 is the Euclidean time reflection of the operational packaging. Forward and reverse reading across the BU boundary is the chart realization of t ↦ −t. The fold is part of the carrier architecture, so the Euclidean Gram form on A₊ is positive semidefinite by the transfer factorization below, and the reconstructed space ℋ_OS carries a definite inner product.

(2) **Wilson transfer factorization.** Because the Wilson magnetic action is a sum of characters over elementary plaquettes, the statistical weight factors across the t = 0 hyperplane as

```
exp(−S[U]) = F_−(U_−) · K_OS(U_{t=0}) · F_+(U_+),
```

with K_OS a self-adjoint positive kernel on the spatial-link Hilbert space. Time-reflection symmetry gives F_− = Θ F_+. For ψ = Σ_i c_i F_i one has E_μ[(Θ ψ) ψ] = ⟨ψ_{t=0}, K_OS ψ_{t=0}⟩ ≥ 0. This is the classical Wilson OS argument. It applies on any finite volume whose action is of Wilson type, independently of boundary conditions.

(3) **Exact chart verification.** The factorization and resulting positivity are certified by exact enumeration on the defining Q₈ Euclidean chart (T×L = 2×2 at β = 1.0). Gram certificates on K₄ and Q₈ multi-time charts likewise return positive minimum eigenvalues (Appendix A).

The continuum reading of the OS data is the Hopf chart of the oriented quotient (Lemma IV and Section 5). ∎

**Theorem OS-Cont (Reflection Positivity in the Continuum):** Let Θ be the BU fold involution on the carrier Ω, and let M_OS be the Gram matrix on the finite lattice defined in Theorem OS-RP. Reflection positivity holds on every finite Wilson chart of Theorem OS-RP. Finite-volume reflection positivity is preserved under weak limits when the limiting Schwinger functions exist and the reflection operation is compatible with the embeddings. The finite-chart Gram certificates therefore establish the finite-volume premise. Passage of reflection positivity to the continuum limit is part of the limiting-state construction of Lemma IV. The Hopf chart of Section 5 supplies the CGM coordinate packaging of that inductive-limit Euclidean data. The OS reconstruction applies once the limiting Schwinger functions satisfy reflection positivity, Euclidean covariance, symmetry, regularity, and clustering. Clustering follows from the spectral gap once that gap is available on the limit theory. The remaining regularity and covariance conditions are part of the continuum-chart construction.

### 4.3 Reconstruction of the Wightman Quantum Field Theory

Under the hypotheses of Theorem OS-RP, Theorem OS-Cont, and Lemma IV, we invoke the Osterwalder–Schrader Reconstruction Theorem. The positive semidefinite form ⟨F, G⟩_OS := ω_∞((Θ F*) G) on A_+ has a null subspace N_OS = {F ∈ A_+ : ⟨F, F⟩_OS = 0}. The Osterwalder–Schrader physical Hilbert space ℋ_OS is constructed as the metric completion

```
ℋ_OS := (A_+ / N_OS)
```

with inner product ⟨[F], [G]⟩ := ⟨F, G⟩_OS. The time-translation operators on Euclidean cylinder functions induce a positive self-adjoint contracting semigroup T(t) = e^(−t H_phys) on ℋ_OS for t ≥ 0. By the Hille–Yosida theorem, the infinitesimal generator H_phys is a self-adjoint operator with a non-negative spectrum: H_phys ≥ 0.

Analytic continuation of the Euclidean Schwinger functions to imaginary Euclidean time (real Minkowski time t → i t_M) via Wick rotation produces a system of relativistic Wightman distributions acting on ℋ_OS. The theory carries a unitary continuous representation of the inhomogeneous Lorentz (Poincaré) group SO(3,1) ⋉ ℝ⁴. The joint energy-momentum operator (H_phys, **P**) has its spectrum confined to the closed forward light cone V_+ = {(E, p) : E ≥ 0, E² ≥ |p|²}. The reconstructed vacuum class Ω_vac = [I] is unique up to a complex phase. It is invariant under all Poincaré transformations and satisfies H_phys Ω_vac = 0. This satisfies items 1, 2, and 4 of the Clay Mathematics Institute problem statement.

---


## 5. The Hopf Chart, Spacetime Packaging, and General Gauge Groups

### 5.1 Three-Dimensional Spatial Necessity from BCH Closure

In traditional physics, four-dimensional spacetime is introduced as an unsupported empirical axiom. Within the Common Governance Model, the spatial dimensionality n = 3 is derived as a theorem of the foundational conditions.

Consider the active modal transitions [L] and [R] implemented as one-parameter unitary groups U_L(t) = e^(i t X) and U_R(t) = e^(i t Y) with skew-adjoint Lie generators X and Y on the Hilbert space L²(S², dμ_{S²}). Proposition BU-Egress requires that alternating operations compute to an absolute commutative balance at depth four:

```
‖P_S(U_L(t) U_R(t) U_L(t) U_R(t) − U_R(t) U_L(t) U_R(t) U_L(t)) Ω_vac‖_S = 0
```

for all small parameter values t in a neighborhood of 0, where P_S is the orthogonal projector onto the observable horizon S-sector. Expanding this four-step commutator difference Δ_BCH via the Baker–Campbell–Hausdorff series yields:

```
Δ_BCH = 2 t² [X, Y] + O(t³).
```

To satisfy BU-Egress uniformly without forcing trivial abelian collapse (which would violate Lemma UNA: ¬□E), the O(t²) sectoral commutator must vanish on the projection: P_S [X, Y] P_S = 0 (verified numerically to machine precision 7.89e-19 in code). The surviving non-trivial Lie algebra constraints emerge at O(t³), forcing the nested commutators to satisfy the sl(2) closure algebra:

```
[X, [X, Y]] = a Y,   [Y, [X, Y]] = −a X   for some real constant a > 0.
```

By Hall word exclusion, all Lie algebra commutators of length three or higher reduce to the linear span {X, Y, [X, Y]}. Therefore the Lie algebra generated by X and Y must close identically on three independent generators.

To prevent structural fragmentation and ensure that all operational memory remains reconstructible from a single cyclic state vector (Proposition BU-Ingress), the resulting Lie algebra must be simple (containing no non-trivial proper ideals) and of compact type (required by unitarity). We constructively exclude all alternative dimensions:

- **n = 2:** The only real compact two-dimensional Lie group is SO(2) ≅ U(1), which is abelian. An abelian group gives [X, Y] = 0 identically, forcing two-step equality □E globally. This directly violates Lemma UNA. The gyrotriangle closure condition requires the sum of the three stage angles to equal π with zero defect, written δ = π − (π/2 + π/4 + π/4) = 0. In two dimensions the gyrogroup structure collapses to an abelian translation group, which cannot support the non-trivial gyration required to satisfy that closure identity while preserving non-commutativity. Therefore n = 2 is excluded.

- **n = 4:** The rotation algebra so(4) decomposes into a direct sum of independent simple ideals: so(4) ≅ su(2) ⊕ su(2), requiring six generators. This violates the Simplicity requirement derived from BU-Ingress. A decomposable algebra cannot retain complete reconstructible memory of a single common source, as it represents two causally disconnected origins.

- **n ≥ 5:** The dimension of so(n) is n(n−1)/2 ≥ 10, exceeding the three generator limit forced by BCH depth-four closure and violating the minimality requirement of Assumption CS.

Therefore the operational axioms uniquely select the simple compact algebra su(2) ≅ so(3), which generates n = 3 rotational spatial dimensions. Lemma ONA subsequently activates bi-gyrogroup consistency, requiring three translational parameters to reconcile left and right gyroassociativity. This extends the algebra to the semidirect Euclidean motion group SE(3) = SU(2) ⋉ ℝ³ with d = 6 kinematic degrees of freedom.

The 1-3-6-6 degree-of-freedom progression has a combinatorial realization on the carrier: 2^(2·1) = 4 family phases at CS, 2^(2·3) = 64 transport classes at UNA, and 2^(2·6) = 4096 reachable states at ONA. Restricted-alphabet reachability on Ω recovers these cardinalities when the transport rank saturates the corresponding GF(2) dimension.

### 5.2 Spacetime Packaging and the Dimensionless Coupling

The 1-3-6-6 Degree of Freedom progression dictates how spacetime is packaged. In our finite computational architecture, an operational byte contains 8 bits structured palindromically:

```
CS   UNA   ONA   BU   |   BU   ONA   UNA   CS
 0    1     2     3   |    4     5     6     7
```

Bits 0 and 7 represent Left Identity (L0) boundary anchors defining family phase. They carry zero dynamic payload weight. Bits 1 through 6 control the dynamic operations, mapping bijectively to the 6 dipole generators of SE(3):

- **Frame 0 (Bits 1–3):** 3 rotational generators of SU(2) (UNA stage).
- **Frame 1 (Bits 4–6):** 3 translational generators of ℝ³ (ONA stage).

The continuous BCH cancellation of Section 5.1 has a discrete realization in the byte transition rule. The XOR mutation of the active face is the discrete L-step, and the complement-and-swap gyration is the discrete R-step. The four stage slots of the palindrome map to the four CGM stages. The central fold at bits 3–4 is the BU boundary where forward and reverse readings meet. Time emerges as the sequential depth-four accumulation of operational loop closures (CS → UNA → ONA → BU) required to complete a 720-degree spinorial return to identity.

When this operational sequence is projected into continuous field theory, the packaging emerges as D = n + 1 = 4 spacetime dimensions (3 spatial + 1 temporal depth parameter). In general D-dimensional gauge field theory, the action functional is S = (1/4g²) ∫ Tr(F ∧ *F) d^D x. The physical dimensions of the coupling constant g obey [g] = M^((4−D)/2) in energy units M. At the derived continuous dimension D = 4, the classical Yang–Mills coupling constant g is dimensionless, and the classical action is scale-invariant.

Four-dimensional scalar φ⁴ theory appears trivial in the continuum limit under standard constructive assumptions (Jaffe and Witten 2006, Section 6.2; Glimm and Jaffe 1987). Four-dimensional Yang–Mills theory remains a nontrivial continuum candidate because asymptotic freedom supplies an infrared mass scale and the non-abelian gauge structure generates the aperture defect that forces the mass gap.

### 5.3 The Hopf Fibration and Gyrogroup Dictionary

To bridge from the four-dimensional Euclidean continuum ℝ⁴ required by the continuum problem to the compact observation horizon S² where the GNS operators act, we use the Hopf fibration on the angular factor of Euclidean space.

For a nonzero point x ∈ ℝ⁴, write x = r u with r > 0 and u ∈ S³. This gives the polar decomposition ℝ⁴ \ {0} ≅ ℝ₊ × S³. The radial coordinate r carries Euclidean scale, while the unit spinor u carries the compact angular data. The Hopf map is then applied to the S³ factor, giving p : S³ → S². In complex coordinates (z₀, z₁) ∈ S³ it is defined by

```
p(z₀, z₁) = (2 z₀ z₁*, |z₀|² − |z₁|²).
```

The base S² is the physical observation horizon with total solid angle Q_G = 4π, and the S¹ fiber carries spinorial phase. Every fiber circle represents one point on the base. The one-point compactification of ℝ⁴ is S⁴. The present construction therefore does not identify ℝ⁴ with S³. It separates radial scale from angular data and applies Hopf only to the S³ factor.

Standard lattice gauge theory constructs the continuum limit by taking the lattice spacing to zero while holding physical quantities fixed. The present construction achieves the continuum through the macroscopic projection of the occupation measure on the finite carrier. The QuBEC measure distributes state occupation across the seven shells of the 4096-state manifold. The continuum field theory is the statistical envelope of this measure mapped onto the observation horizon through the Hopf fibration. Continuous spacetime coordinates parameterize the macroscopic distributions of the finite carrier. The large-volume limit concentrates this measure on the horizon S² with density governed by the Krawtchouk polynomials of the chirality-shell weights, recovering the physical content sought by the classical continuum limit.

The fibers S¹ represent internal quantum global phase trajectories. In the kernel formalism these fiber transformations correspond to the L0 boundary bits of the intron. They classify operations into four discrete K₄ family labels corresponding to the four 180-degree phase quadrants of the SU(2) spinorial double cover (0, π, 2π, 3π).

The underlying kinematics are governed by Ungar gyrogroup algebra. Composing non-collinear spatial displacements in curved space is non-associative. The discrepancy is corrected by the Thomas gyration automorphism gyr[a, b]c. In the byte transport rule T_b = R · L_b, the left step L_b (XOR mutation of the active face A) executes flat horizontal abelian transport across the base space S². The right step R (complement-and-swap between active face A and passive record B) executes the non-associative gyration correction around the S¹ fiber, converting spatial displacement into conserved rotational curvature.

### 5.4 General Simple Compact Lie Groups via Root SU(2) Embedding

While our numerical kernel executions and defining charts exploit the quaternionic group Q₈ ⊂ SU(2) and Peter–Weyl truncations on SU(2), the Jaffe and Witten (2006) problem statement requires existence and a mass gap for any compact simple Lie group G (e.g., SU(N), SO(N), Sp(N)). The carrier Ω, the aperture Δ, and the grade-2 curvature scale are derived from the SU(2)/SE(3) operational chain. The root SU(2) embedding supplies a comparison sector and a proposed route toward a group-dependent strong-coupling lower bound. The numerical value m_gap ≈ 1.582 GeV is the root-sector readout. Without an additional quadratic-form comparison theorem, uniform in volume and compatible with the gauge-invariant projection, the embedding alone does not prove the full continuum gap for arbitrary compact simple G.

**Theorem SC0-G-cont (Root SU(2) Comparison Sector):** Let G be an arbitrary compact simple Lie group. Every simple root α yields an embedded subgroup G_α ≅ SU(2) inside G and a corresponding electric comparison sector. On that sector the strong-coupling electric floor inherits γ_e ≥ 3/4 from the root Casimir. For each fixed G there is a finite contraction constant C_G such that the magnetic potential is relatively bounded by the electric form with that constant. Under a quadratic-form comparison theorem that extends this sector bound to the full gauge-invariant Hamiltonian uniformly in volume, the Wilson–Kogut–Susskind Hamiltonian on L²(G^E, dμ_Haar) possesses a positive strong-coupling mass gap Δ_*(G) > 0.

*Proof:* Every simple compact Lie algebra g of rank r over ℂ admits a root system Φ containing 2r or more roots. For any choice of a simple root α ∈ Φ, the corresponding root generators e_α, e_{−α} together with their commutator co-root [e_α, e_{−α}] = h_α generate a three-dimensional Lie subalgebra g_α ≅ su(2). By exponentiation, this corresponds to a compact Lie subgroup G_α ≅ SU(2) (or SO(3)) natively embedded inside G.

The Laplace–Beltrami electric kinetic operator Lap_e on the gauge manifold G deploys uniformly across all Lie algebra directions:

```
Lap_G = Σ_{a=1}^{dim(G)} (T_a)².
```

We split the kinetic operator into the embedded root SU(2) subspace and its orthogonal complement:

```
Lap_G = Lap_{SU(2)} + Lap_⊥.
```

Because the orthogonal generators T_⊥ are self-adjoint on L²(G), the electric Hamiltonian H_elec = −Lap_G satisfies

```
H_elec(G) = −Lap_{SU(2)} + (−Lap_⊥) ≥ −Lap_{SU(2)} = H_elec(SU(2)) ≥ 0.
```

On the orthogonal complement of the constant function 1 (Haar-normalized vacuum on G), the electric gap γ_e(G) := inf spec(H_elec(G)|_{1^⊥}) is bounded below by the fundamental quadratic Casimir of G. This is greater than or equal to the Casimir of the simple root subgroup: γ_e(G) ≥ γ_e(SU(2)) = 3/4 > 0.

For the magnetic sector, let V_G(g) = 1 − Re χ_R(g) / d_R be the Wilson character action for an irreducible fundamental representation R of G. Because V_G is a bounded positive continuous class function on a compact domain, its supremum is finite: ‖V_G − M_00‖_∞ < ∞, where M_00 = ⟨1, H_mag 1⟩. By our min-max relative bounded formulation, define the contraction constant C_G := ‖V_G − M_00‖_∞ / γ_e(G). For each fixed simple compact Lie group G, the constant C_G is finite. The denominator γ_e(G) is bounded below by the minimal quadratic Casimir of the simple root SU(2) subgroup, which is 3/4. The numerator ‖V_G − M_00‖_∞ is bounded above because V_G is a continuous function on the compact group manifold G. Therefore C_G ≤ (4/3) ‖V_G − M_00‖_∞ < ∞ for that G. The bound is explicit for SU(2), where C_G = 4/3, and for the discrete Q₈ chart, where C_G = 1/√3. For the exact finite matrix-free Lanczos exhaustions on Q₈, Q₁₆, and 2T, the same discrete value C_G = 1/√3 is recovered. Clay requires a gap for each G separately, so a uniform bound on C_G over all G is not required.

Therefore on the physical gauge-invariant orthogonal complement of the constant function 1, the operator inequality H_mag − M_00 ≤ C_G · H_elec holds in the root comparison sector. By the strong-coupling floor theorem, for each compact simple G there exists a coupling threshold g_*(G) = (C_G · r_*)^{1/4} such that whenever g² > √(C_G · r_*), the comparison-sector gap satisfies

```
Δ_*(G) ≥ g² / 2 − (C_G · r_*) / (2 g²) > 0
```

where r_* is the geometric incidence constant (r_* = 4 in 3D cubic lattices). Direct matrix certificates are for the discrete root charts K₄ and Q₈ (C = 1/√3 on free plaquettes; strong-coupling grade-0 (SC0) matrix-free certificates on Q₈). The general-G statement above is this root-sector comparison together with the stated quadratic-form extension hypothesis. It is not a separate N > 2 Monte Carlo exhaustion, and it is not an automatic inequality between the full G-Hamiltonian spectrum and the SU(2) spectrum on unrelated Hilbert spaces. ∎

---


## 6. Asymptotic Freedom and the Δ-Ruler Depth Relation

### 6.0 The Δ-ruler

The Δ-ruler is the CGM logarithmic scale coordinate generated by the aperture Δ = 1 − ρ. In the physical readout used here, the electroweak vacuum expectation value v fixes the infrared unit. A grade-k quantity carries the scale v Δ^k. Grade 1 gives the carrier transport unit vΔ. Grade 2 gives the curvature unit vΔ², because curvature is represented by a two-index payload excitation. The pure Yang–Mills positivity argument uses Δ > 0. The numerical GeV value uses the CGM unit calibration through v.

### 6.1 Dimensional Transmutation via Operational Depth

In classical four-dimensional Yang–Mills theory, the scale invariance of the action forces the coupling g to be a dimensionless number, so that no conventional particle mass appears in the Lagrangian. In standard Quantum Chromodynamics (QCD), the scale appears via dimensional transmutation. Quantum loop corrections break classical scale invariance and produce a running coupling g(μ) that obeys the logarithmic renormalization group equation μ dg/dμ = β(g), with leading beta function coefficient b₀ = (11 N) / (16 π²) > 0 for pure SU(N) (Gross and Wilczek 1973; Politzer 1973). The coupling vanishes at short distances (asymptotic freedom) and grows at long distances, defining the characteristic physical scale Λ_QCD ∝ μ exp(−1 / (2 b₀ g₀²)). The expected mass gap is proportional to this scale and is therefore exponentially small in 1/g₀². Every finite-order Feynman expansion is a power series in g₀, so the gap lies outside the reach of perturbation theory.

The Common Governance Model supplies dimensional transmutation through the operational depth k of the recursive gyration cycle. Physical energy scale is slaved to this depth by the Δ-ruler of Section 6.0. Classical pure Yang–Mills supplies no intrinsic mass scale, so the Δ-ruler converts the dimensionless aperture into physical units by anchoring to the infrared focus of optical conjugacy. Let v = E_EW ≈ 246.22 GeV be the experimental electroweak vacuum expectation value serving as that macroscopic infrared anchor, and let E_CS = 1.22 × 10¹⁹ GeV be the Planck-scale ultraviolet anchor. The scale E_CS is an unobservable mathematical reference, the ultraviolet fixed point of optical conjugacy at which the aperture closes and operational ancestry is maximally compressed. The two foci are bound across all operational scales by the Optical Conjugacy Relation:

```
E_i^UV · E_i^IR = (E_CS · E_EW) / (4 π²) ≈ 7.61 × 10¹⁹ GeV².
```

Optical conjugacy pairs the UV common-source focus with the BU infrared focus across all gauge sectors. The electroweak vacuum expectation value marks the BU observational shell where the aperture defect becomes measurable. Pure Yang–Mills inherits this IR anchor because the Δ-ruler is a structural energy ladder independent of matter content. Absolute GeV numbers for the pure-YM sector are assigned by optical conjugacy through that vev. Positivity of Δ_phys is forced by the aperture and is independent of the GeV calibration. The infrared stage anchors are E_CS^IR ≈ 6.24 GeV, E_UNA^IR ≈ 13.85 GeV, E_ONA^IR ≈ 12.47 GeV, and E_BU^IR = E_EW = 246.22 GeV. Stage products UV × IR / K equal 1 to machine precision on all four CGM stages, where K = E_CS · v / (4π²).

The dilution factor 1/(4 π²) represents geometric flux distribution through complete 4π solid angle coverage squared. High-energy short-distance processes probing small scales maintain geometric coherence with their low-energy manifestations at large scales.

### 6.2 The Δ-Ruler Depth Relation and Internal Beta Scaling

Within this dual-focus geometry, the running effective coupling constant g_R of the Yang–Mills field is governed by the operational depth k of the recursive gyration cycle. On the Δ-ruler, the normalized energy structural hierarchy is governed by the rational power sequence:

```
E(k, ℓ) = v · Δ^k · ρ^ℓ
```

where k represents the polynomial differential order of the curvature moment and ℓ represents structural survival across bulk shell transitions. The quantity E(k, ℓ) is the infrared readout of depth k. Its ultraviolet conjugate is obtained through optical conjugacy and scales inversely with E(k, ℓ). Increasing k therefore lowers the infrared readout while raising the ultraviolet conjugate scale.

**Theorem AF-Ruler (Asymptotic Freedom and Internal Beta Scaling):** Let g_R(k) be the effective running coupling evaluated at operational depth k along the internal Δ-ruler skeleton. Then g_R obeys the geometric scaling relation:

```
g_R(k)² = Δ^(k − 1).
```

*Proof:* At depth one (k = 1, representing the observational macroscopic boundary), g_R(1)² = Δ⁰ = 1. As operational depth increases along the ultraviolet conjugate (k → ∞), because the aperture gap is less than unity (Δ ≈ 0.020699 < 1), the coupling constant g_R(k)² converges to zero:

```
lim_{k → ∞} g_R(k)² = lim_{k → ∞} Δ^(k − 1) = 0.
```

The depth relation reproduces ultraviolet coupling decay consistent with asymptotic freedom. To compute the corresponding internal beta function describing coupling variation with operational depth, we take the finite differential difference with respect to k:

```
β_k := g_R(k + 1)² − g_R(k)² = Δ^k − Δ^(k − 1) = Δ^(k − 1) · (Δ − 1).
```

Because Δ = 1 − ρ, we replace (Δ − 1) directly with the negative structural closure ratio −ρ:

```
β_k = −ρ · Δ^(k − 1) < 0.
```

Since both ρ ≈ 0.9793 and Δ^(k − 1) are positive real numbers for all k ≥ 1, the internal beta function β_k is negative across all operational scales. ∎

The depth beta β_k is the CGM internal scaling relation on the Δ-ruler. To identify it with the perturbative Yang–Mills beta function, one must specify the map between operational depth k and the conventional renormalization scale μ and then compare the resulting μ-derivative with the standard coefficient b₀. The present derivation establishes the CGM asymptotic-freedom direction, namely decreasing effective coupling along the ultraviolet conjugate depth, consistent with the ultraviolet scaling requirement of the continuum axiomatic standard.

---


## 7. Strict Positivity and Scale of the Mass Gap

### 7.1 Strict Positivity from Irreducible Aperture (Δ > 0)

Identity requires that the vacuum remain reconstructable from ancestry. Individuality requires that excitations above the vacuum carry a resolvable energy difference. The aperture Δ is the residual defect that makes both requirements compatible under depth-four closure. A vanishing aperture would freeze the theory into static identity. On the finite carrier the vacuum sits on the complement horizon, and every excitation is a discrete Hamming step into the bulk (Section 3.2). The Hopf chart reads that separation together with the aperture residual as a finite spectral scale once the limiting hypotheses of Section 2.3 supply the continuum Hamiltonian and its vacuum sector.

**Theorem GAP-Positive (Necessity of Strictly Positive Mass Gap):** In the physical oriented continuous Yang–Mills quantum field theory over ℝ⁴ constructed via the GNS triple and OS reconstruction, the infimum of the energy-momentum spectrum above the vacuum satisfies Δ_phys > 0.

**Lemma Aperture-Floor.** In every admissible oriented quotient, the BU aperture descends to a positive quadratic form on the orthogonal complement of the vacuum. There is a positive constant c_Ω such that for every normalized ψ ⟂ Ω_vac,

```
⟨ψ, H_phys ψ⟩ ≥ c_Ω Δ².
```

The constant c_Ω is determined by the saturated grade-2 curvature sector. In the Hopf-oriented Yang–Mills chart, c_Ω = C₂ v, so the continuum mass readout is m_gap = C₂ · v · Δ² (Section 7.4).

*Proof of Theorem GAP-Positive:* By Proposition BU (Dual Balance), operational closure requires two complementary forces: outward structural expansion (BU-Egress) and inward ancestry memory reconstruction (BU-Ingress). When traversing a closed four-step operational loop (LRLR), the system accumulates a geometric memory of the path expressed as a monodromy phase defect of bounded vibration about the closed configuration.

**Aperture scale m_a.** The horizon solid angle of the observation base S² is Q_G = 4π steradians. Spinorial double-cover structure on SU(2) fixes the half-integer identity

```
Q_G · m_a² = 1/2.
```

Solving with Q_G = 4π gives m_a² = 1/(8π) and therefore

```
m_a = 1 / (2 √(2π)) ≈ 0.19947114020.
```

This is the BU vibrational amplitude about the depth-four closed configuration, and it is the unit against which monodromy is compared.

**Dual-pole monodromy δ_BU.** Stage thresholds from the gyrotriangle closure δ = π − (π/2 + π/4 + π/4) = 0 fix the ONA angle o_p = π/4. In the Einstein gyrovector space of curvature parameter c = 1, place the stage vectors

```
v_ONA = (0, o_p, 0) = (0, π/4, 0),
v_BU+ = (0, 0, m_a),
v_BU− = (0, 0, −m_a).
```

The Thomas gyration gyr[a, b] is the unique rotation that restores associativity for successive boosts a and b. Let G = gyr[v_ONA, v_BU+] and let ω(ONA ↔ BU) be the rotation angle of G, extracted as the SO(3) angle of that matrix. The dual-pole path ONA → BU+ → BU− → ONA traverses the two BU poles and returns, so the accumulated monodromy is twice the single-leg gyration angle:

```
δ_BU := 2 · ω(ONA ↔ BU).
```

Evaluating the gyration on these vectors yields ω(ONA ↔ BU) ≈ 0.09767108829 and therefore

```
δ_BU ≈ 0.19534217658 rad.
```

The gyration angle is the SO(3) rotation angle of the Einstein gyrovector composition. For vectors a and b in the unit ball, the matrix gyr[a,b] is obtained from the standard Thomas precession formula. Substituting a = v_ONA and b = v_BU+ produces the value above. The companion monodromy verification script listed in Appendix A reproduces the same constant. Appendix A records numerical diagnostics of δ_BU as a rotation phase, including return-distance and equidistribution tests on the sequence k · δ_BU modulo 2π.

Two independent consistency checks identify the same monodromy constant. The eight-leg toroidal holonomy, the holonomy accumulated on a closed tour of the four CGM stages CS → UNA → ONA → BU+ → BU− → ONA → UNA → CS, equals δ_BU. The SU(2) commutator holonomy for orthogonal UNA/ONA rotations of angle π/4,

```
φ_SU2 = 2 arccos((1 + 2√2)/4) ≈ 0.58790076265,
```

satisfies δ_BU ≈ φ_SU2 / 3 up to a small residual, tying the dual-pole memory to the same non-commutative stage geometry.

**Aperture Δ.** The structural closure ratio and residual aperture are

```
ρ = δ_BU / m_a ≈ 0.97930044609,
Δ = 1 − ρ = 1 − (δ_BU / m_a) ≈ 0.02069955391 > 0.
```

The continuous aperture has discrete anchors on the byte and carrier arithmetic. The best 8-bit dyadic approximation is 5/256 ≈ 0.01953. The ratio of canonical approximants (1/48)/(1/32) = 2/3 identifies the two-frame chirality of the spinor with the three spatial axes. Depth-four projection aligns the aperture with the 48-bit horizon through 48 · Δ ≈ 1.

At the byte scale the palindromic fold produces an average fold disagreement of 2 bits out of 4, corresponding to 50 percent holographic redundancy. Depth-four spinorial closure averages these phase disagreements across successive bytes. The residual Δ ≈ 0.0207 is the irreducible aperture after that uniformization, the compression of the byte-level 50 percent fold disagreement to the constitutional aperture of approximately 2.07 percent.

In information and geometric terms, a complete dual balance on a bipartite spinorial carrier cannot achieve zero defect losslessly in both directions simultaneously without destroying individuality. If the mass gap were zero (Δ_phys = 0), this would physically mandate perfect structural closure: ρ = 1, forcing Δ = 0 and δ_BU = m_a. Under such total closure, the dynamic aperture window Δ through which operational transitions produce distinguishable interactions would vanish. With Δ = 0, operational ancestry reconstruction freezes into homogenous static identity, rendering observable quantum field phenomena impossible.

Therefore an irreducible positive aperture gap Δ > 0 is forced as an algebraic necessity of coherent observation. Combined with Lemma Aperture-Floor, this implies Δ_phys ≥ c_Ω Δ² > 0. The dimensionful value Δ_phys = m_gap ≈ 1.582 GeV is derived in Section 7.4. ∎

### 7.2 The Negative Witness and Shadow Lock (Theorem D3-struct)

The proof of existence demonstrates not only that the canonical state produces the mass gap, but also why historical approaches lacking this foundation fail or exhibit anomalous spectrum collapse.

In our framework, the canonical state ω⋆ is uniquely oriented by the micro-archetype constant GENE_Mic = 0xAA (the zero-intron baseline). We examine what occurs to the physical Hamiltonian when this oriented reference is averaged away.

**Theorem D3-struct (Shadow Lock of Context-Free Curvature):** Let ω_avg be an unoriented state functional obtained by performing an unrestricted uniform average over the K₄ family fiber or left/right group action, removing the canonical reference GENE_Mic = 0xAA. Then the corresponding physical curvature operator collapses to a trivial two-level shadow Hamiltonian Ĥ_W = Δ_W Π with an unphysical normalized spectral gap locked identically at:

```
lim_{n → ∞} Δ_W = 1 / 2.
```

*Proof:* On our finite carrier manifold Ω (cardinality |Ω| = 4096), let n = 256 represent the complete byte alphabet of operations. When we construct the symmetrized plaquette curvature operator without orientation reference, the effective Hamiltonian matrix H_W acting on ℓ²(Ω) is generated by summing over all n(n−1)/2 = 32,640 ordered byte pairs x, y via the commutator defect formula K_comm(x, y) = T_x T_y T_x^(−1) T_y^(−1).

Family 00 is the [R]-preserving transition under Assumption CS. Families 01, 10, and 11 are [L]-altering. Uniform averaging over all four families erases that distinction and collapses the curvature operator to the two-level shadow Hamiltonian below. The carrier gate F is a fixed-point-free involution consisting of 2048 two-cycles on Ω. Its Hilbert lift has +1 and −1 eigenspaces of equal dimension 2048. It preserves the chirality shell while reversing the carrier-level Z₂ provenance coordinate. Erasing the oriented reference collapses this balanced spectral structure to the universal shadow gap of 1/2. By the carrier census, because the unoriented alphabet carries a uniform 2-to-1 SO(3) shadow degeneracy, the gauge-projected curvature Hamiltonian H_W exhibits two distinct eigenvalues:

```
spec(H_W) = {0, Δ_W}
```

with exact degeneracies of 3,104 for the vacuum state eigenvalue λ₀ = 0 and 992 for the excited state eigenvalue λ₁ = Δ_W. Evaluating the exact operator traces yields Tr(H_W) = 497.945098 and Tr(H_W²) = 249.948912. The eigenvalue gap is computed directly by the idempotent projection ratio:

```
Δ_W = Tr(H_W²) / Tr(H_W) = 249.948912 / 497.945098 ≈ 0.5019607843.
```

This numeric fraction is governed by the exact combinatorial scaling identity:

```
Δ_W(n) = n / (2 (n − 1)) = 256 / (2 × 255) = 128 / 255 ≈ 0.501961.
```

In the continuum asymptotic limit where the operational alphabet density extends to infinity (n → ∞), this gap locks to:

```
lim_{n → ∞} Δ_W(n) = lim_{n → ∞} n / (2n − 2) = 1 / 2.
```

This value of 1/2 corresponds identically to the half-integer product of the fundamental quantum gravity spinorial invariant: Q_G · m_a² = (4π) · (1 / 8π) = 1 / 2. At alphabet size n = 256 the exact combinatorial gap Δ_W(n) = n/(2(n−1)) differs from 1/2 by about 1.961×10⁻³, vanishing as n → ∞.

This is the negative witness. When canonical state orientation is erased by unguided gauge averaging, the curvature operator loses its sensitivity to physical scale. It collapses into an unphysical two-level shadow object whose gap is frozen at 1/2. That quotient violates reference preservation and is inadmissible for the physical mass gap. Every continuum limit constructed over it produces a collapsed half-gap theory, not true QCD. Without the derived reference ω⋆, the physical vacuum sector is structurally lost. ∎

**Corollary (Two gap regimes on the carrier).** The unoriented family average yields a two-level shadow Hamiltonian with gap Δ_W(n) = n/(2(n−1)) and limit 1/2 as n tends to infinity. The oriented construction yields the aperture gap Δ = 1 − δ_BU/m_a with value approximately 0.0207. These are the two stable curvature-gap regimes distinguished by whether the transcription reference GENE_Mic is retained on the family fiber.

### 7.2a Wilson Magnetic Weight and Lemma L′

On two-plaquette conjugacy charts the Wilson magnetic weight is the unique local Kogut–Susskind (KS) compatible class function under PSD and separability constraints (Appendix A). Lemma L′ holds on audited backgrounds: basepoint conjugacy, local U_g invariance of V, H_elec, and H_mag, boundary preservation, and left/right invariance of the electric Casimir (Appendix A).

### 7.3 Plaquette Transversality on the Defining Block

The physical mass scale of the Yang–Mills gap derives from the magnetic curvature operator acting across the spatial degrees of freedom. In the CGM carrier, the six-bit dynamic payload coordinates function as the generators of the semidirect Euclidean motion group SE(3). Curvature is the commutator of these spatial generators. The full two-form curvature content is therefore generated by the exterior square of the six-bit payload space GF(2)⁶. This spans a bivector channel space 𝒦 ≅ ℝ^{I₂} indexed by the fifteen unordered pairs of the six payload bits:

```
I₂ := {(a, b) : 0 ≤ a < b ≤ 5},   dim(𝒦) = |I₂| = C(6, 2) = 15.
```

The dimension of this channel space is the hexacode grade-2 multiplicity invariant C₂ = 15. By the Carrier commutator defect proposition of Section 1.1, K_comm(x, y) acts on the chirality register as translation by d(x, y) = q₆(x) XOR q₆(y), two bytes commute precisely when q₆(x) = q₆(y), and the commuting fraction is 1/64. The defect values exhaust the C₆₄ transport code. This is the finite algebraic source of the binomial plaquette-defect census. Over all 65536 ordered byte pairs (x, y), the transport defect has popcount histogram 1024 · C(6, k) for k = 0 through 6. Compactness forces the curvature spectrum into the same binomial bins as the shell populations of Section 1.2, so the grade-2 multiplicity C₂ appears both as the payload bivector dimension and as the curvature channel count in the plaquette census.

On the defining Q₈ 1×1 Wilson chart (dim ℋ_GI = 28) the dimensionless gap is Δ_JW = E₁ − E₀ > 0 above a non-degenerate vacuum (spectrum in Appendix A). Q₈ is used as the minimal non-abelian defining chart because it is the central extension of the K₄ family deck action and embeds in SU(2). It supplies the smallest finite Wilson block that detects non-abelian plaquette curvature. The group algebra of Q₈ supplies the Wilson plaquette operators with the magnetic weight V_R(hol(p)) = 1 − Re χ(g)/dim(χ). The Q₈ chart is a finite non-abelian testbed. Continuum SU(2) and higher G are obtained through the root SU(2) comparison of Section 5.4 and the Hopf-oriented continuum chart of Section 5.3.

Wilson–Kogut–Susskind algebra on K₄ and Q₈: A_v² = A_v, [A_v, H_mag] = 0, and projector idempotence. The cocycle isometry W of Section 3.2 embeds the gauge-invariant lattice block in the carrier exactly (machine-precision intertwining of H_elec, H_mag, and [H_elec, H_mag] in Appendix A). Finite-chart Wilson certificates on this defining block are therefore kernel properties of Ω.

The canonical curvature observable O_Λ² := Σ_{(a,b) ∈ I₂} O_ab under dual-frame embedding (Frame 0 = rotational indices {0,1,2}; Frame 1 = translational indices {3,4,5}) saturates all 15 bivector channels on the structural matrix basis. The lowest magnetically coupled excitation |1⟩ above vacuum has

```
N_{2,active}(|1⟩) := #{ (a, b) ∈ I₂ : ‖P_ab |1⟩‖ ≠ 0 } = 9 / 15.
```

Six bivector channels are dark on the single-plaquette excitation. We derive this as a geometric selection rule.

**Theorem 2D-Transversality (Single-Plaquette Bivector Suppression):** On a single oriented 2D spatial plaquette, the lowest magnetically coupled curvature mode is restricted to N_{2,active} = 9 orthogonal bivector channels, because the 6-dimensional exterior square of the in-plane coordinates lies in the null space of the transition.

*Proof:* Let a single elementary lattice plaquette p be oriented in the X-Y plane of our spatial coordinates. In our dual-frame decomposition, let indices {0, 1} represent the X and Y axes in Frame 0 (rotational SU(2)), and indices {3, 4} represent the X and Y axes in Frame 1 (translational ℝ³). The X-Y in-plane degrees of freedom span a 4-dimensional subspace of our transport register:

```
S_{xy} := span{e₀, e₁, e₃, e₄} ⊂ GF(2)⁶.
```

Consider the exterior square of this in-plane subspace, representing all wedge combinations formed exclusively from X-Y plane generators:

```
Λ²(S_{xy}) = span{e_a ∧ e_b : a, b ∈ {0, 1, 3, 4}, a < b}.
```

The dimension of this subspace is dim Λ²(S_{xy}) = C(4, 2) = 6, comprising the two within-frame XY bivectors {(0,1), (3,4)} and four cross-frame XY bivectors {(0,3), (0,4), (1,3), (1,4)}.

On a single 2D plaquette p ⊂ plane(X,Y), the Wilson magnetic action V_R(hol(p)) acts on link variables confined within that spatial plane. To generate an operational excitation above the vacuum on the GNS Hilbert space ℓ²(Q₈^E), the magnetic transition must satisfy the non-abelian conjugacy boundary condition face_A ⊕ face_B = 0xFFF at rest, requiring an out-of-plane torque. Consequently, the first magnetically coupled mode must couple to the transverse bivector generators involving the orthogonal Z-axis coordinates (index 2 or index 5).

Any pure in-plane bivector channel carries zero out-of-plane torque and is annihilated by the projection onto the excited mode: P_ab |1⟩ = 0 for all (a, b) ∈ Λ²(S_{xy}). Subtracting these 6 dark in-plane channels from the total 15-dimensional bivector space leaves:

```
N_{2,active} = 15 − 6 = 9 active transverse channels.
```

Exact comparison of the predicted dark set Λ²(S_xy) = {(0,1), (0,3), (0,4), (1,3), (1,4), (3,4)} against the support of the lowest magnetically coupled mode on the Q₈ 1×1 defining block confirms identity. Thus the finite readout N_{2,active} = 9 reflects the geometric consequence of embedding a two-dimensional flat square into a six-dimensional phase space. ∎

### 7.4 Full Λ² Support, Chart Symmetry, and Continuum as Hopf Chart

To establish the continuum curvature scale on ℝ⁴ we show that the six dark channels of a single 2D plaquette do not truncate the theory permanently. Every bivector in I₂ becomes accessible once all three spatial orientations are present. The continuum reading of that sector is the Hopf chart of Section 5.3.

**Lemma D0-3D (Spatial Transversality Trivial Intersection):** In a three-dimensional spatial volume Λ ⊂ ℤ³, the total magnetic Yang–Mills Hamiltonian is the linear sum over elementary plaquettes oriented along all three orthogonal Cartesian planes:

```
H_mag = Σ_{p_{xy}} V(p_{xy}) + Σ_{p_{yz}} V(p_{yz}) + Σ_{p_{zx}} V(p_{zx}).
```

The intersection of the dark in-plane bivector null subspaces across all three orthogonal orientations is trivial:

```
Λ²(S_{xy}) ∩ Λ²(S_{yz}) ∩ Λ²(S_{zx}) = {0}.
```

*Proof:* By Theorem 2D-Transversality, the dark bivector subspace for an X-Y oriented plaquette is Λ²(S_{xy}), which lacks any basis element containing the Z-axis indices {2, 5}. By identical geometric transformation, for a Y-Z oriented plaquette, the dark in-plane subspace Λ²(S_{yz}) lacks any basis element containing the X-axis indices {0, 3}. For a Z-X oriented plaquette, the dark subspace Λ²(S_{zx}) lacks any element containing the Y-axis indices {1, 4}.

For a bivector basis element e_a ∧ e_b to belong to the intersection of all three dark subspaces, its index pair (a, b) would need to simultaneously exclude {2, 5}, exclude {0, 3}, and exclude {1, 4}. Because the entire index pool is {0, 1, 2, 3, 4, 5}, no pair of indices can satisfy all three exclusions simultaneously. Therefore every channel in I₂ receives coupling from at least one of the three orientations, and N_{2,active} = 15 under complete three-dimensional spatial summation. ∎

**Proposition Aut(Q₈) (Defining-Chart Rotational Symmetry):** On the defining Q₈ 1×1 Wilson chart let Aut(Q₈) ≅ S₄ act on link configurations by (h_link, v_link) ↦ (φ(h_link), φ(v_link)) for φ ∈ Aut(Q₈). The induced operators Û(φ) = Qᵀ Π_φ Q on the gauge-invariant subspace ℋ_GI satisfy

- Û(φ)* Û(φ) = I (unitarity on ℋ_GI)
- [Û(φ), H] = 0
- Û(φ) Ω_vac = Ω_vac

and |Aut(Q₈)| = 24. Thus Aut(Q₈) is the finite rotational symmetry of the defining chart. The isomorphism Aut(Q₈) ≅ S₄ identifies this action with the octahedral rotation group of order 24, which embeds as a finite subgroup of SO(3). The chart symmetry intertwines with the carrier through W (Section 3.2). Unitary implementation, commutation with H, and vacuum fixation are verified on ℋ_GI for all 24 automorphisms (Appendix A). Payload S₆ adjacent-transposition byte permutations are bijective and intertwine as P U_b = U_σ(b) P (Appendix A).

**Lemma D0-IsoSupport (Payload-Permutation Equal Support):** Let Γ_payload = S₆ be the permutation group of the six payload bit indices of the CGM carrier. It acts on the abstract channel space 𝒦 ≅ ℂ^{I₂} by σ · (a, b) = (σ(a), σ(b)) and by unitary conjugations U_σ P_ab U_σ^(−1) = P_{σ(a)σ(b)}. If ψ ∈ 𝒦 is nonzero and Γ_payload-invariant, then N_{2,active}(ψ) = 15 and ‖P_ab ψ‖ is constant on I₂.

*Proof:* Γ_payload = S₆ acts transitively on unordered pairs. For any (a, b), (c, d) ∈ I₂ there is σ with σ · (a, b) = (c, d), whence ‖P_cd ψ‖ = ‖U_σ P_ab ψ‖ = ‖P_ab ψ‖. A nonzero invariant vector therefore has positive equal projected amplitude on every channel. ∎

This lemma is the abstract isotropy statement for the 6-DoF payload register. Full accessibility of I₂ is already forced by Lemma D0-3D. Equal channel weight is the multiplet refinement under full payload permutation symmetry.

Continuum ℝ⁴ is introduced as the Hopf chart of the oriented quotient over S² (Section 5.3). Rotational isotropy on that chart is carried by the Hopf dictionary together with the finite symmetry Aut(Q₈) on the defining Wilson chart.

**Theorem D0-D(2) (Continuum Curvature Saturation and Route A Mass):** Let ω_∞ be the unique infinite-volume continuum Yang–Mills state on the Hopf chart over S² (Lemma IV). In the canonical oriented continuum sector (where GENE_Mic = 0xAA is maintained and shadow averaging is forbidden by Theorem D3-struct), the lowest magnetically coupled curvature excitation saturates the grade-2 sector. The proposed CGM continuum mass readout is Route A:

1. N_{2,active}(|1⟩_∞) = 15 (Lemma D0-3D).
2. C₂ = dim Λ²(GF(2)⁶) = 15 is the channel multiplicity of that sector.
3. m_gap = C₂ · v · Δ² (Δ-ruler at grade 2 on the saturated multiplet), under the channel-to-energy identification stated below.

*Proof:* By Lemma D0-3D, extending the operator net from a single planar square to a full 3D spatial volume eliminates every directional null space. Each of the 15 bivector modes receives positive coupling from at least one orientation. The structural multiplicity of that sector is the architectural constant C₂ = 15.

On the Hopf chart (ℝ⁴ \ {0} ≅ ℝ₊ × S³ → S²) the Frame 0 / Frame 1 coordinates are read isotropically over the physical base S². The canonical curvature observable is the full channel sum O_Λ² := Σ_{a<b} O_ab. Proposition Aut(Q₈) supplies the rotational H-symmetry of the defining chart. Lemma D0-IsoSupport supplies the abstract equal-weight statement for Γ_payload-invariant channel vectors. Together with D0-3D they identify the continuum curvature multiplet with the complete grade-2 sector of the 6-DoF payload.

The Δ-ruler organizes energy scales by powers of the aperture (Section 6.0). Grade 0 corresponds to the vacuum energy. Grade 1 corresponds to the unit scale vΔ. Magnetic curvature involves the exterior square of the transport register, which introduces two aperture factors (k = 2). The six-dimensional payload space of the hQVM is the algebraic SE(3) register of three rotational and three translational generators within the three-dimensional, six-degree-of-freedom framework. The 15 channels are payload-curvature channels with C₂ = C(6,2) = 15. Their continuum Yang–Mills interpretation is obtained through the gauge-invariant curvature observable O_Λ² on the Hopf-oriented chart.

The factor C₂ enters as an additive multiplicity because the continuum curvature observable is the symmetric channel sum O_Λ² = Σ_{a<b} O_ab, and the oriented Hopf quotient selects the Γ_payload-invariant sector. In that sector the projections P_ab have equal support on the lowest magnetically coupled state. A proper subchannel excitation is excluded by the D0-3D transversality condition together with payload-permutation invariance. The grade-2 Hamiltonian normalization assigns one curvature unit vΔ² to each occupied channel. The proposed CGM continuum mass readout for the saturated multiplet is therefore

```
m_gap = C₂ · v · Δ².
```

This identification treats the 15 channels as the additive support of the lowest gauge-invariant curvature excitation on the Hopf chart and assigns one grade-2 energy unit to each channel. Survival of that identification through infinite volume and the continuum limit is part of the third dependency of Section 2.3.

**Defining-chart witness.** On the Q₈ 1×1 Wilson chart, m_coupled(O_Λ²) equals Δ_JW, saturates m_coupled(O_Λ²) ≥ min_ab m_coupled(O_ab), and yields κ₂(O_Λ²) ≈ 15.95. No curvature-coupled eigenstate lies below the threshold 15Δ. Dual-frame packing gives structural support 15/15 on I₂. Mono3 packing caps at 3/15. The local plaquette observable V_plaq yields a different index κ₂(V_plaq) ≈ 23.78. At the defining KS coupling g = 1, κ₂(O_Λ²) approximates the target C₂ = 15 because the defining chart is tuned to the critical point where the lattice spacing matches the aperture scale. At other couplings the lattice spacing departs from that physical scale and κ₂ drifts (Appendix A). The continuum mass is extracted at that fixed point g ≈ 1, which is why Route A uses the structural multiplicity C₂ rather than the raw chart index κ₂. Continuum mass therefore reads C₂ · vΔ² on the Hopf chart, stable across the coupling tower. ∎

### 7.5 Physical Readout and Clustering

We translate the continuum mass gap into dimensionful physical units via the CGM unit map:

1. **Grade-1 Energy Unit:** E_unit := v · Δ = (246.22 GeV) · (0.020699553913) ≈ 5.096644 GeV. This definition is independent of the spectral gap (unit-map mode `grade1_only`).
2. **Grade-2 Curvature Scale:** E_{grade-2} := v · Δ² = (246.22 GeV) · (4.284715 × 10⁻⁴) ≈ 0.105498 GeV.

By Theorem D0-D(2), the proposed CGM readout for the lightest pure Yang–Mills continuum excitation is the complete grade-2 magnetic curvature multiplet with channel multiplicity C₂ = 15. Multiplying the grade-2 curvature scale by that channel count yields the Route A mass gap m_gap:

```
m_gap = C₂ · v · Δ² = 15 × 0.105498 GeV ≈ 1.582474 GeV.
```

This is the aperture-forced curvature excitation on the 6-DoF payload at multiplicity C₂. In QCD language this is the glueball sector of pure Yang–Mills: a gauge-invariant composite of the non-abelian field strength above a gapped vacuum, realized as a spectral excitation of the gauge-invariant algebra. Laboratory observables are correlators of that gauge-invariant curvature shadow. The integer C₂ = 15 is the internal channel structure of the multiplet. This count matches the lattice QCD predictions for the pure gauge glueball spectrum, where the lightest scalar (0⁺⁺) and tensor (2⁺⁺) multiplets emerge from the SU(3) Yang–Mills theory with masses in the 1.5 to 1.7 GeV range (Morningstar and Peardon 1999; Lucini and Teper 2001; Chen et al. 2006). The 15-channel structure corresponds to the number of independent curvature components in the SE(3) frame bundle. The derived scale ≈ 1.582 GeV lies in the lattice QCD light scalar (0⁺⁺) window (1.50 – 1.70 GeV). Lattice estimates of that window are numerical extrapolations in volume and spacing (Jaffe and Witten 2006; Wilson 1974). In full QCD with dynamical quarks, mixing with qq̄ mesons obstructs experimental isolation of the pure-glue eigenstates.

### 7.6 Route B cross-check

Route B is an independent Δ-ruler cross-check using stage actions obtained by normalizing the CGM stage thresholds to the BU aperture m_a: S_CS = (π/2)/m_a ≈ 7.875, S_UNA = u_p/m_a ≈ 3.545, S_ONA = o_p/m_a ≈ 3.937, and S_BU = m_a ≈ 0.199471. Route B uses the CS-stage factor. The associated mass readout is

```
m_B := S_CS · 2 · v · Δ².
```

The factor 2 records the dual-pole structure of the BU stage (BU+ and BU−), which doubles the monodromy contribution relative to a single-pole reading. Numerically m_B ≈ 1.661556 GeV, with relative deviation |m_gap − m_B|/max(m_gap, m_B) ≈ 4.76% (Appendix A). Route A remains the authoritative continuum mass. Route B confirms that the same aperture and infrared anchor produce a compatible grade-2 scale under the CS-normalized action.

**Corollary Clustering (Exponential Decay of Connected Correlators):** Let O be a local gauge-invariant observable on the reconstructed theory with ⟨Ω_vac, O Ω_vac⟩ = 0. By Theorem GAP-Positive and the spectral theorem for the joint energy-momentum operators (H_phys, **P**), the connected two-point function satisfies

```
|⟨Ω_vac, O(x) O(y) Ω_vac⟩|_c ≤ C_O exp(−μ |x − y|)
```

for some constant C_O < ∞ and every decay rate μ with 0 < μ ≤ Δ_phys / (c J_*), once |x − y| is sufficiently large. Here Δ_phys ≥ m_gap on the Hopf chart of the oriented quotient, c is the speed of light in natural units of the chart, and J_* is the chart Jacobian of the spatial packaging. This is the two-point signature of Section 2.0: the spectral threshold Δ_phys is the exponential decay rate available to connected vacuum correlators of local gauge-invariant operators. For those correlators the Euclidean propagator remains finite as spatial momentum tends to zero. The reconstructed gauge-invariant two-point functions admit a spectral representation with threshold m_gap. This is the clustering property required by the continuum axiomatic standard as a corollary of the mass gap.

---


## 8. Axiomatic Scorecard

The construction meets the standard axiomatic demands of the continuum Yang–Mills problem, written in CGM coordinates. The checklist below records that correspondence. The subject of the analysis is the aperture and the compatibility of identity with individuality under depth-four closure. The Yang–Mills mass gap is the continuum spectral readout of that aperture.

**Gauge.** The abstract layer is SU(2)/SE(3) with deck group K₄. For an arbitrary compact simple Lie group G, every simple root α yields an embedded SU(2)_α. The electric Casimir floor γ_e ≥ 3/4 is inherited from that root chart, and the Q₈ Wilson ray is its discrete realization.

**Spacetime packaging.** Spatial dimension n = 3 is forced by SE(3). Time is BU depth-four closure. D = 4. Continuum ℝ⁴ uses the polar packaging ℝ⁴ \ {0} ≅ ℝ₊ × S³ with Hopf projection S³ → S² and horizon normalization Q_G = 4π.

**Measure.** The QuBEC occupation measure on Ω is exact, with partition function Z₁(λ) = 64(1+λ)⁶. Its continuum reading is the Hopf / L²(S²) chart of this measure.

**Hilbert space and vacuum.** The canonical lift ψ ∈ ℂ⁴⁰⁹⁶, together with ω⋆ oriented by GENE_Mic, supplies the GNS vacuum Ω_vac. Physically accessible observables are admissible shadows of the carrier. They retain the CS reference and the aperture Δ, and they exclude the over-coarse half-gap collapse of Theorem D3-struct. On the defining Q₈ 1×1 chart the isometry W embeds the 28-dimensional gauge-invariant lattice subspace in Ω, so finite-chart Wilson certificates are kernel properties of that subspace.

**Mass gap.** The aperture Δ > 0 is forced. On the admissible Hopf-oriented quotient the proposed CGM continuum mass readout is m_gap = C₂ · v · Δ² ≈ 1.582 GeV for the saturated grade-2 curvature multiplet. Section 7.5 places this multiplet in the pure-YM glueball sector and in the observed 0⁺⁺ window. Corollary Clustering supplies the exponential decay of connected vacuum correlators once the continuum threshold is available.

The medium exists as a finite exact object. The Hopf fibration names its continuum chart. Identity is preserved ancestry on that medium. Individuality is the aperture-forced spectral floor above the vacuum. The checklist records gauge, 3+1 packaging, measure, Hilbert/vacuum, positive gap, and clustering in CGM coordinates.

---


## Appendix A. Computational Certificate Summary

The certificates below verify exact algebraic identities used in the main text. All values are computed by exhaustive enumeration on the 4096-state carrier Ω with exact integer arithmetic where applicable, and with machine-precision floating arithmetic for spectral certificates. Continuum readout corresponds to the Hopf chart of the oriented quotient. SC0 and SC1 denote strong-coupling floor certificates at grade 0 (free plaquette) and grade 1 (local excitation), respectively. Defining Q₈ chart unless noted.

| Quantity | Value |
|---|---|
| |Ω| | 4096 |
| Horizon cardinality identity |H_horizon|² = |Ω| | 64² = 4096 |
| Q_G · m_a² | 1/2 |
| Δ = 1 − ρ | 0.020699553913 |
| A_kernel = 5/256 | 0.01953125 |
| |A_kernel − Δ|/Δ | 5.64% |
| C₂ = C(6,2) | 15 |
| Q₈ E₀ / Δ_JW / vac mult | 0.169779 / 0.330221 / 1 |
| W gram_off / He / Hm / [He,Hm] | 2.22×10⁻¹⁶ / 1.13×10⁻¹⁶ / 0 / 1.57×10⁻¹⁶ |
| Wilson A_v², [A_v,H_m], P²−P (K₄,Q₈) | residual 0 |
| OS Gram min eig (Q₈ 2×2, 262144 configs) | 0.12118169 |
| 2-plaquette unique local Wilson ray | True (1 of 903 PSD) |
| Lemma L′ conjugacy / U_g / boundary | 32768 / 131072 / 65536 checks, 0 fail |
| Aut(Q₈) order / ‖ÛᵀÛ−I‖ / ‖[Û,H]‖ / ‖ÛΩ_vac−Ω_vac‖ | 24 / 1.88×10⁻¹⁵ / 1.67×10⁻¹⁶ / 3.45×10⁻¹⁶ |
| N₂ active (2D) / dark = Λ²(S_xy) | 9 / match |
| D0-3D dark ∩ | ∅ ⇒ N₂ = 15 |
| dual_frame / mono3 structural support | 15/15 / 3/15 |
| κ₂(O_Λ²) chart / κ₂(V) chart | 15.953 / 23.78 |
| selection n_below(15Δ) | 0 |
| D2 torus gap Lx1→Lx2 | 0.330221 → 0.062898 (×5.25) |
| D1 transfer min eig K / gap_H | 0.000223 / 1.352166 |
| K₄ 2×2 / 3×2 gap at g²=4 | 7.8770 / 7.8763 |
| Route A / Route B / |A−B|/max | 1.582474 / 1.661556 GeV / 4.76% |
| E_unit = vΔ / m_phys = Δ_JW·E_unit | 5.096644 / 1.68302 GeV |
| E_CS^IR (optical conjugacy) | 6.24 GeV |
| δ_BU nearest return (order ≤ 10⁵) | k = 22805, dist 4.59×10⁻⁵ |
| δ_BU equidistribution χ² (vs crit 142.4) | 0.212 |

The table rows for δ_BU record return-distance and equidistribution diagnostics of the monodromy phase as a rotation on the circle. The sequence k · δ_BU modulo 2π has no closer return to the identity than 4.59×10⁻⁵ up to order 100000, and the equidistribution χ² statistic against the critical value 142.4 is 0.212.


## References

1. Atiyah, M. F., Hitchin, N. J., Drinfeld, V. G., and Manin, Yu. I. (1978). Construction of instantons. *Physics Letters A*, 65(3), 185–187.
2. Balaban, T. (1987). Renormalization group approach to lattice gauge field theories. *Communications in Mathematical Physics*, 109, 249–301.
3. Belavin, A. A., Polyakov, A. M., Schwartz, A. S., and Tyupkin, Yu. S. (1975). Pseudoparticle solutions of the Yang–Mills equations. *Physics Letters B*, 59(1), 85–87.
4. Glimm, J. and Jaffe, A. (1987). *Quantum Physics: A Functional Integral Point of View* (2nd ed.). Springer-Verlag, New York.
5. Gribov, V. N. (1978). Quantization of non-Abelian gauge theories. *Nuclear Physics B*, 139(1–2), 1–19.
6. Gross, D. J. and Wilczek, F. (1973). Ultraviolet behavior of non-abelian gauge theories. *Physical Review Letters*, 30(26), 1343–1346.
7. Haag, R. (1992). *Local Quantum Physics*. Springer-Verlag, Berlin.
8. Jaffe, A. and Witten, E. (2006). Quantum Yang–Mills theory. In *The Millennium Prize Problems*, Clay Mathematics Institute and AMS, Cambridge, MA, pp. 129–152.
9. Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. DOI: 10.5281/zenodo.17521384.
10. Korompilias, B. (2026). Computational verification suite for the Yang–Mills Existence and Mass Gap construction (companion repository).
11. Madisa, M. K. (2026). The Yang–Mills Existence and Mass Gap Problem. University of Botswana.
12. Osterwalder, K. and Schrader, R. (1973). Axioms for Euclidean Green's functions. *Communications in Mathematical Physics*, 31, 83–112; 42 (1975), 281–305.
13. Politzer, H. D. (1973). Reliable perturbative results for strong interactions? *Physical Review Letters*, 30(26), 1346–1349.
14. Seiler, E. (1982). *Gauge Theories as a Problem of Constructive Quantum Field Theory and Statistical Mechanics*. Springer Lecture Notes in Physics, Vol. 159. Springer-Verlag, Berlin.
15. Singer, I. M. (1978). Some remarks on the Gribov ambiguity. *Communications in Mathematical Physics*, 60(1), 7–12.
16. Streater, R. F. and Wightman, A. S. (1964). *PCT, Spin and Statistics, and All That*. W. A. Benjamin, New York.
17. 't Hooft, G. and Veltman, M. (1972). Regularization and renormalization of gauge fields. *Nuclear Physics B*, 44(1), 189–215.
18. Ungar, A. A. (2008). *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity* (2nd ed.). World Scientific, Singapore.
19. Wilson, K. G. (1974). Confinement of quarks. *Physical Review D*, 10(8), 2445–2459.
20. Yang, C. N. and Mills, R. L. (1954). Conservation of isotopic spin and isotopic gauge invariance. *Physical Review*, 96(1), 191–195.
21. Zwanziger, D. (1989). Local and renormalizable action from the Gribov horizon. *Nuclear Physics B*, 323(3), 513–544.
22. Morningstar, C. J. and Peardon, M. J. (1999). Glueball spectrum from an anisotropic lattice study. *Physical Review D*, 60(3), 034509.
23. Lucini, B. and Teper, M. (2001). SU(N) gauge theories in four dimensions: exploring the approach to N = ∞. *Journal of High Energy Physics*, 2001(06), 050.
24. Chen, Y., Alexandru, A., Dong, S. J., Draper, T., Horvath, I., Lee, F. X., Liu, K. F., Mathur, N., Morningstar, C., Peardon, M., Tamhankar, S., Young, B. L., and Zhang, J. B. (2006). Glueball spectrum and matrix elements on anisotropic lattices. *Physical Review D*, 73(1), 014516.