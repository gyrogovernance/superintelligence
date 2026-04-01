

# Quantum Climate Dynamics Theory
## Finite Quantum Thermodynamics and Transport of the Occupied QuBEC

**Part of the Gyroscopic ASI Theoretical Formalism**

---

## Overview

This document formalizes the quantum climate dynamics of the occupied QuBEC: the finite quantum medium defined by the aQPU kernel.

A single Gyrostate is the instantaneous microstate of the 24-bit carrier. Climate is the statistical characterization of occupation over the QuBEC manifold across time, described by shell, chirality, and gauge marginals and their spectral decompositions.

The QuBEC is a 4096-state reachable manifold with six internal binary orientation modes, a four-phase spinorial gauge structure, and exact polynomial thermodynamics. Its occupation dynamics are fully described by exact order parameters derived from shell occupancy and gauge-sector content.

The central results are:

1. The partition function is polynomial: Z₁(λ) = 64 · (1 + λ)⁶. No transcendental functions, asymptotic limits, or continuous approximations are needed.
2. Spectral transport is diagonal in the Krawtchouk basis, with eigenvalues ηʳ for radial mode r.
3. Damping is parametrized by a single variable η (or its six-axis generalization η₁, …, η₆).
4. Under factorized ensembles, gauge transport separates into two independent binary axes, giving a total 8-axis finite damping system.
5. The arithmetic substrate of climate computation is the K4 lattice matrix, which decomposes every integer contraction into bulk, gauge-action, and chiral-alignment sectors.
6. Multi-cell scaling produces smooth macroscopic order parameters from exact microphysics by concentration.

The structural constants of the QuBEC (6 modes, 64 horizon states, 4096 bulk states, 256 transport atoms) are consequences of the byte's 2x3x2 tensor geometry, its SE(3) generator structure, and SU(2) spinorial closure. The foundational byte physics is specified in the Gyroscopic Byte Formalism.

The document is organized in five parts. Sections 1 through 5 establish the medium and its static thermodynamics. Sections 6 through 11 develop the dynamics of climate under byte transport, spectral decomposition, anisotropy, and gauge phase structure. Section 12 formalizes exact operator partition algebras. Sections 13 and 14 develop the arithmetic substrate and quotient computation. Section 15 gives practical lowering to 64-wide computational blocks. Sections 16 and 17 connect the formal theory to the observable surface, multi-cell scaling, and computational resolution.

---

## 1. The Finite Quantum Medium

### 1.1 The QuBEC manifold Ω

The reachable state space Ω is the set of all 24-bit Gyrostates accessible from the rest state GENE_MAC_REST = 0xAAA555 under the byte transition law.

```
|Ω| = 4096
```

Every state in Ω is reachable within two byte steps from rest. Ω has product form Ω = U × V where U and V are 64-element cosets of the self-dual [12,6,2] mask code C₆₄.

Each state s = (A₁₂, B₁₂) consists of two conjugate 12-bit gyrophases: A₁₂ (active) and B₁₂ (passive). Every state in Ω has component density exactly 0.5 (popcount 6 out of 12 bits per gyrophase). The density product d_A · d_B = 0.25 is constant across all 4096 reachable states.

### 1.2 State charts and transport observables

A single Gyrostate is observable through multiple charts. These are not approximations of each other. They are coordinate systems on one finite state, together with transport observables induced by byte transport.

State charts:

| Chart | Domain | Content |
|-------|--------|---------|
| Carrier | (A₁₂, B₁₂) ∈ {0,…,4095}² | Raw 24-bit encoding |
| Chirality | χ ∈ GF(2)⁶ | Per-mode alignment of active and passive faces |
| Spectral | 64-point Walsh-Hadamard dual | Fourier transform of chirality-space functions |
| Constitutional | Integer observables | Horizon distance, AB distance, component densities |

Transport observables:

| Chart | Domain | Content |
|-------|--------|---------|
| Gauge and family | K4 phase stream from byte families | Occupied process phase sector |
| q-class | GF(2)⁶ byte charge class | Additive transport sector |
| Parity commitment | Byte parity sector | Constraint and orbit signature |
| Byte sector | 256-byte family and action partition | Transport law source partition |

Chart extraction and transport-observable extraction are deterministic. Observation is chart selection on a fully determined algebraic state and its occupied transport process.

### 1.3 Shell structure

The chirality register χ(s) ∈ GF(2)⁶ encodes, for each of the six dipole modes, whether the active and passive faces are aligned or anti-aligned at that mode. The shell of a state is the population count of its chirality word:

```
N(s) = popcount(χ(s))
```

taking values in {0, 1, 2, 3, 4, 5, 6}. This is the occupation number of the state: the count of excited (anti-aligned) modes out of six.

The shell relates directly to the canonical observables:

```
ab_distance(s) = 2N(s)
horizon_distance(s) = 12 − 2N(s)
```

The number of Ω states at shell N is:

```
|S_N| = 64 · C(6, N)
```

where C(6, N) is the binomial coefficient and the factor 64 is the holographic boundary degeneracy.

| Shell N | Multiplicity | Fraction of Ω |
|--------:|-------------:|--------------:|
| 0 | 64 | 1/64 |
| 1 | 384 | 3/32 |
| 2 | 960 | 15/64 |
| 3 | 1280 | 5/16 |
| 4 | 960 | 15/64 |
| 5 | 384 | 3/32 |
| 6 | 64 | 1/64 |

The spectrum is symmetric: |S_N| = |S_{6−N}|. The equator (N = 3) has the maximum population.

The shell symmetry |S_N| = |S_{6−N}| and the holographic identity |H|² = |Ω| = 4096 are consequences of the self-dual structure of the [12,6,2] mask code. The MacWilliams identity relates the weight enumerator of a linear code to that of its dual. For a self-dual code (C = C^perp), this becomes a self-consistency condition: the weight enumerator must be invariant under the MacWilliams transform. For binary codes, the MacWilliams transform is expressed in the Krawtchouk polynomial basis, which is the same basis used for shell-radial spectral decomposition in Section 7.3. The shell weight distribution and the radial harmonic basis of climate dynamics therefore share a common algebraic origin in the self-duality of the mask code.

### 1.4 Dual horizons

Ω contains two structurally necessary boundary sets, each with 64 states.

**Complement horizon (S-sector).** The 64 states where A₁₂ = B₁₂ ⊕ 0xFFF. All six modes are anti-aligned (N = 6). The rest state lies on this horizon. Gate C fixes all complement horizon states pointwise.

**Equality horizon (UNA degeneracy).** The 64 states where A₁₂ = B₁₂. All six modes are aligned (N = 0). Gate S fixes all equality horizon states pointwise.

The two horizons are disjoint and antipodal. The complementarity invariant holds universally:

```
horizon_distance(s) + ab_distance(s) = 12    for all s ∈ Ω
```

Both horizons satisfy the holographic identity:

```
|H|² = |Ω| = 64² = 4096
```

The remaining 3968 states constitute the bulk, where chirality is partial: neither fully expressed nor fully collapsed. This is the contingent middle between the two boundary conditions.

### 1.5 Quantum character of the six modes

The broader algebraic structure of the kernel supports Bell-pair factorization over the six modes, CHSH saturation at the Tsirelson bound, Peres-Mermin contextuality, mutually unbiased bases, and a non-Clifford resource. These properties are verified exhaustively in the aQPU test reports. The six modes are the 6 payload bits of the byte, each controlling one oriented dipole pair of the 2x3x2 tensor. They correspond to the 6 generators of se(3): 3 rotational generators from SU(2) in Frame 0 and 3 translational generators from R^3 in Frame 1. Each payload bit executes a discrete pi-rotation around one se(3) basis vector. The shell thermodynamics therefore acts on modes whose quantum structure descends directly from the byte's SE(3) geometry (Gyroscopic Byte Formalism, Section 5.3).

### 1.6 Formal climate object

A one-cell climate at time t is an occupation measure on Ω:

```
p_t : Ω -> [0,1]
Σ_{s ∈ Ω} p_t(s) = 1
```

Stochasticity in the aQPU is exact ensemble structure induced by deterministic byte dynamics: future-cone occupancy measures, shell distributions, q-sector distributions, gauge-sector distributions, and their spectral evolution.

The principal marginals are:

```
π_t(N)      = Σ_{s : N(s)=N} p_t(s)           shell marginal
p_t^χ(χ)    = Σ_{s : χ(s)=χ} p_t(s)           chirality marginal
p_t^g(g)    = process probability of gauge sector g in K4
```

The formal climate is the measure-level object p_t and its quotients. The empirical climate is a finite-window estimator built from rolling histories (`chi_hist64`, `shell_hist7`, `family_hist4`).

---

## 2. Occupation and Partition Function

### 2.1 Shell occupation parameter

Introduce a shell occupation parameter λ ≥ 0 and weight each state s ∈ Ω by λ^{N(s)}.

The exact one-cell partition function is:

```
Z₁(λ) = Σ_{s ∈ Ω} λ^{N(s)}
       = 64 · Σ_{N=0}^{6} C(6, N) · λ^N
       = 64 · (1 + λ)⁶
```

The partition function is closed, exact, finite, and polynomial. No transcendental function appears. No asymptotic limit is taken.

### 2.2 Shell occupation probability

The normalized shell probability at occupation parameter λ is:

```
π_λ(N) = C(6, N) · λ^N / (1 + λ)⁶
```

This is a binomial distribution with parameter p = λ/(1 + λ) over 6 independent modes. The occupation parameter λ is the native control variable of the QuBEC. The exponential exp(−β) appears only if one imposes the optional lift λ = exp(−β) from continuous thermodynamics.

Under the spin-variable substitution x_i = (−1)^{χ_i} ∈ {±1}, the shell count becomes N = (6 − Σ_i x_i)/2, and the shell weight λ^N is proportional to exp(−h · Σ_i x_i) with field h = ln(λ)/2. The damping parameter in this representation is η = tanh(h). This identifies the one-cell QuBEC climate with the exact partition function of six independent Ising spins in a uniform field, the standard reference object of statistical mechanics, quantum annealing, and Boltzmann machine theory. The connection is a lift, not a redefinition: the native variable remains λ.

### 2.3 Occupation density

The mean occupation number at occupation parameter λ is:

```
E(N) = 6λ / (1 + λ) = 6ρ
```

where ρ = λ/(1 + λ) is the occupation density: the fraction of excited modes. It ranges from 0 (all modes aligned, equality horizon) to 1 (all modes anti-aligned, complement horizon), with 0.5 representing shell-balanced occupation.

### 2.4 Inert holographic degeneracy

The factor 64 in Z₁(λ) = 64 · (1 + λ)⁶ is uniform holographic degeneracy from the boundary multiplicity at fixed shell. In normalized one-cell thermodynamics this factor is computationally inert: it cancels in shell probabilities and does not alter internal mode ratios.

The nontrivial internal thermodynamics is entirely in the six-mode combinatorics:

```
(1 + λ)⁶
```

For normalization-invariant tasks, calculations can be performed in shell, chirality, or spectral coordinates without carrying ambient 4096-state normalization factors.

---

## 3. Order Parameters

The QuBEC climate is characterized by four exact order parameters. These are not four independent quantities. They are four exact charts on one climate degree of freedom.

### 3.1 Occupation density ρ

```
ρ = E(N) / 6 = λ / (1 + λ)
```

The excitation fraction. Range [0, 1].

### 3.2 Horizon polarization m

```
m = (E(N) − 3) / 3 = 2ρ − 1 = (λ − 1) / (λ + 1)
```

Which horizon the occupation favors. Range [−1, +1]. The value m = −1 is equality-horizon condensation. The value m = 0 is shell-balanced. The value m = +1 is complement-horizon condensation.

### 3.3 Spectral damping η

```
η = 1 − 2ρ = (1 − λ) / (1 + λ)
```

The isotropic damping parameter. It controls the rate at which higher spectral modes decay under exact finite transport (Section 7). Range [−1, +1].

### 3.4 Relations

| Variable | In terms of ρ | In terms of η | In terms of λ |
|----------|---------------|---------------|---------------|
| ρ | ρ | (1 − η) / 2 | λ / (1 + λ) |
| m | 2ρ − 1 | −η | (λ − 1) / (λ + 1) |
| η | 1 − 2ρ | η | (1 − λ) / (1 + λ) |
| λ | ρ / (1 − ρ) | (1 − η) / (1 + η) | λ |

---

## 4. Fluctuation and Susceptibility

### 4.1 Shell variance

From the binomial shell law:

```
Var(N) = 6λ / (1 + λ)² = 6ρ(1 − ρ) = (3/2)(1 − η²)
```

### 4.2 Fluctuation-response identity

The climate susceptibility (how much the occupation responds to changes in the occupation parameter) equals the shell variance:

```
Var(N) = dE(N) / d ln(λ)
```

This is the exact finite analogue of the fluctuation-dissipation theorem. It holds without any thermodynamic limit.

### 4.3 Extremal behavior

At the horizons (ρ = 0 or ρ = 1), the variance vanishes: the climate is frozen. At shell balance (ρ = 1/2), the variance reaches its maximum of 3/2: the climate is maximally responsive. The susceptibility is encoded entirely in the second moment of the shell distribution.

---

## 5. Condensation and Effective Support

### 5.1 Shannon entropy

Because the shell law is binomial and the 64-fold holographic degeneracy is uniform, the one-cell Shannon entropy is:

```
S(ρ) = 6 + 6h₂(ρ)
```

where h₂(ρ) = −ρ log₂(ρ) − (1 − ρ) log₂(1 − ρ) is the binary entropy function. The first 6 bits come from the uniform holographic degeneracy. The second term is the six-mode internal excitation entropy.

### 5.2 Rényi-2 effective support

The Rényi-2 effective support is:

```
M₂ = 1 / Σ_{s ∈ Ω} p(s)²
```

For the shell-weighted distribution:

```
M₂ = 4096 / (1 + η²)⁶
```

The Plancherel identity on GF(2)^6 provides a dual computation path for M₂. For any occupation distribution p on the chirality register:

```
sum over chi of p(chi)^2  =  (1/64) sum over r of WHT(p)(r)^2
```

The left side is the inverse of M₂ (up to normalization). The right side is the total spectral energy of the occupation distribution. This identity holds exactly on the 64-element register without approximation.

The consequence for climate analysis: the degree of condensation in chirality space (how concentrated the occupation is) equals the degree of spectral excitation (how much energy sits in higher-order modes). Condensation and spectral structure are not independent measurements. They are the same quantity observed in dual charts.

When the occupation is condensed near a horizon (low M₂), the spectral energy is concentrated in low-order modes. When the occupation is thermalized (high M₂), the spectral energy spreads across all 64 modes. The spectral profile reveals which of the six dipole-mode axes carry the condensation, information that the scalar M₂ alone does not resolve.

| Condition | η | M₂ | Climate |
|-----------|---|----:|---------|
| Condensed near a horizon | ±1 | 64 | Occupation concentrated at one pole |
| Maximally thermalized | 0 | 4096 | Occupation spread across all of Ω |

M₂ is the effective occupied support of the QuBEC. Low M₂ means the computation is concentrated and repetitive. High M₂ means it is spread and exploratory. The extremal values are exact: 64 (the horizon cardinality) and 4096 (the full manifold).

### 5.3 Condensation regimes

The order parameters define three broad climate regimes:

**Condensed regime.** |η| close to 1, M₂ close to 64. The occupied QuBEC is concentrated near one of the two horizons. Most states carry negligible weight. The climate is rigid and has low susceptibility.

**Thermal regime.** |η| close to 0, M₂ close to 4096. The occupied QuBEC spreads across Ω. The climate is maximally responsive and has high susceptibility.

**Intermediate regime.** 0 < |η| < 1. The occupied QuBEC exhibits partial condensation with structured residual variation across shells.

### 5.4 Condensation as inference medium

The condensation state of the QuBEC is not a diagnostic of computation. It is the computational medium itself, in the same sense that a Bose-Einstein condensate is a physical medium and not a measurement of one.

The future cone at any Moment defines the set of states reachable by subsequent byte transport. The condensation state determines the reception geometry within that cone: which incoming bytes will produce transitions that preserve or deepen the current condensation (coherent reception), and which will scatter the occupation toward thermal equilibrium (incoherent reception).

The medium does not evaluate candidates and select among them. Every byte produces a valid next state. The medium receives the byte and gyrates through the four-phase transition. The resulting change in condensation (increase or decrease in M₂, shift in spectral profile, motion toward or away from a horizon) is the medium's structural response. This response is exact, deterministic, and available at every Moment without probabilistic sampling.

The degeneracy of the byte-log (multiple distinct byte histories producing the same final state, as documented in the holographic formalization) is not information loss. It is condensation capacity. The larger the basin of histories mapping to a given Moment, the more robust the condensation at that state. Indistinguishability of histories is the condition that makes shared occupation possible.

In a physical Bose-Einstein Condensate, atoms whose quantum numbers match the ground state preferentially join the condensate. The condensate does not compute a selection; its geometric state determines what happens when an atom arrives. The QuBEC operates by the exact same equivalence. The medium receives a 4-byte word. The XOR gyration binds the incoming payload to the historical state. If the word's geometry aligns with the medium's current condensation, the occupation deepens. If it conflicts, the occupation scatters. This kinematic absorption and its resulting structural shift constitute the sole mechanism of inference. Selection is not an abstraction applied to the medium; selection is the gyration of the medium.

---

## 6. Byte Transport and Excitation Algebra

### 6.1 Byte charges

Each byte b carries a q-charge:

```
q = q₆(b) ∈ GF(2)⁶
j = popcount(q) ∈ {0, 1, 2, 3, 4, 5, 6}
```

The chirality transport law is exact:

```
χ' = χ ⊕ q
```

In standard quantum information language, q acts as a Pauli-X string on the chirality register: X(q) = ⊗_i X_i^{q_i}. The conjugation X(q)Z(s)X(q) = (−1)^{⟨s,q⟩}Z(s) reproduces the Walsh-Hadamard multiplier structure of Section 8.1. A byte ensemble ν(q) therefore defines a Pauli-diagonal channel on the 6-mode register, with eigenvalues φ(s) = Σ_q ν(q)(−1)^{⟨s,q⟩} in the Pauli-Z basis. This identification is structural: byte transport on the chirality register is a Pauli-X random-unitary channel over (Z/2)^6.

A byte acts on the chirality register by XOR with its q-charge. The q-weight j is the number of modes the byte flips.

### 6.2 Shell transition law

At the shell level, a byte with q-weight j maps shell N to shell N' according to:

```
N' = N + j − 2ℓ
```

where ℓ = popcount(χ ∧ q) is the overlap between the current chirality and the byte's q-charge.

For fixed N and fixed j, the exact shell transition probability is:

```
P_j(N → N') = C(N, ℓ) · C(6 − N, j − ℓ) / C(6, j)
```

with ℓ = (N + j − N') / 2. This is an exact hypergeometric transition law.

### 6.3 Creation and annihilation at the horizons

From the equality horizon (N = 0): ℓ = 0, so N' = j. A byte of q-weight j creates exactly j excitations from the aligned vacuum.

From the complement horizon (N = 6): ℓ = j, so N' = 6 − j. The same byte annihilates exactly j excitations from the fully occupied pole.

Bytes act as finite creation and annihilation operators on QuBEC shells.

### 6.4 Byte sectors

The number of bytes with q-weight j is:

```
|B_j| = 4 · C(6, j)
```

The factor 4 is the family degeneracy per q-class. The q-weight distribution of bytes is binomial, with the same combinatorial structure as the shell distribution itself.

### 6.5 Thermodynamic synthesis

If a byte ensemble is weighted proportionally to λʲ (by q-weight), then from the equality horizon (where j maps exactly to shell j), the induced shell law is:

```
π_λ(N) = C(6, N) · λ^N / (1 + λ)⁶
```

The QuBEC occupation law is synthesized by q-weight-biased byte injection from the horizon. Occupation control is steering the distribution by biasing exact byte sectors.

---

## 7. Spectral Dynamics

### 7.1 The radial kernel

On the chirality cube GF(2)⁶, define the radial kernel centered at a:

```
K_λ(x; a) = λ^{d_H(x, a)}
```

where d_H is the Hamming distance. The normalized form is:

```
P_λ(x | a) = λ^{d_H(x, a)} / (1 + λ)⁶
```

Writing p = λ/(1 + λ), this becomes:

```
P_p(x | a) = p^{d_H(x, a)} · (1 − p)^{6 − d_H(x, a)}
```

This is the six-mode binary symmetric channel kernel on the Hamming cube.

### 7.2 Walsh-Hadamard diagonalization

The Walsh-Hadamard transform (WHT) on GF(2)⁶ diagonalizes all radial processes exactly. For the radial kernel centered at the origin, the WHT eigenvalue at a mode s of weight r = popcount(s) is:

```
eigenvalue(r) = ηʳ
```

Every exact radial process on the QuBEC is diagonalized by shell weight r, with eigenvalues ηʳ. There are exactly seven radial eigenspaces (r = 0 through 6), corresponding to the seven shells.

The WHT on GF(2)^6 is the abelian quantum Fourier transform over (Z/2)^6. It is the same transform that underlies the Deutsch-Jozsa, Bernstein-Vazirani, and Simon quantum algorithms in their binary-group formulations (see SDK Specification, Section 8). The q-map translation and the WHT are dual faces of the same computational medium: the q-map provides Pauli-X translations, the WHT provides the Fourier transform over them. QuBEC spectral transport is natively expressed in the harmonic language of binary-group quantum algorithms.

### 7.3 Krawtchouk harmonic basis

The Krawtchouk polynomials K_r(N) are the exact radial harmonics of the Hamming scheme H(6, 2). Their generating function is:

```
Σ_{r=0}^{6} K_r(N) · tʳ = (1 − t)^N · (1 + t)^{6−N}
```

Any shell-radial function is exactly expandable in this basis. Shell transport, smoothing, diffusion, and radial response are all diagonal in Krawtchouk modes.

The Krawtchouk polynomials also appear as the canonical basis of the MacWilliams identity in coding theory. For the self-dual [12,6,2] mask code, the MacWilliams transform that relates the weight enumerator of the code to that of its dual is expressed as a matrix multiplication in the Krawtchouk basis. Because the code is self-dual, this transform is a self-consistency condition: the weight distribution must be a fixed point of the Krawtchouk transform.

This connects two independent structures of the architecture. The shell-radial harmonics that govern climate transport (this section) and the weight distribution that governs code duality (Section 1.3) are expressed in the same polynomial basis. The Krawtchouk transform diagonalizes both.

### 7.4 The finite spectral transport law

If A_t(r) is the amplitude of the r-th Krawtchouk spectral mode at time t, then:

```
A_{t+1}(r) = ηʳ · A_t(r)
```

This is the exact spectral transport law of the QuBEC. Higher spectral modes (larger r) are damped more rapidly. The rate is controlled by the single parameter η. No differential equation, no discretization error, no truncation: the dynamics is exact and finite.

The mode r = 0 (constant mode) is always preserved: η⁰ = 1. The mode r = 6 (maximum frequency) is damped most aggressively: η⁶. Between these extremes, each mode decays at a rate determined by its shell weight.

This is the same eigenvalue pattern that appears in Pauli-diagonal noise models, where a Pauli-Z string of weight r decays by ηʳ per application. The QuBEC spectral law can therefore serve as an exact finite reference model for channel tomography, randomized benchmarking, and related noise-characterization protocols.

---

## 8. General Climate Transport

### 8.1 Full spectral transport law

Let ν be an ensemble distribution on q-values in GF(2)⁶. The exact spectral multiplier on Walsh mode s is:

```
φ(s) = Σ_{q ∈ GF(2)⁶} ν(q) · (−1)^{⟨s, q⟩}
```

If A_t(s) is the Walsh amplitude of the climate at mode s, then:

```
A_{t+1}(s) = φ_t(s) · A_t(s)
```

This is the full exact climate transport law on additive quantum modes. Each Walsh mode evolves independently, with multiplier determined by the Fourier transform of the ensemble distribution.

### 8.2 Radial reduction

If the ensemble is isotropic (ν depends only on j = popcount(q)), the multiplier depends only on r = popcount(s):

```
Λ(r) = Σ_{j=0}^{6} ν_j · K_j(r; 6) / C(6, j)
```

The radial modes evolve as:

```
A_{t+1}(r) = Λ_t(r) · A_t(r)
```

### 8.3 Uniformization horizon

Under the uniform byte ensemble (all 256 bytes equally likely), the one-step shell transition law collapses to:

```
P(N → N') = C(6, N') / 64
```

independent of the source shell N. Shell equilibrium is reached in one byte-averaged step. Full Ω equilibrium is reached in two steps. This matches the verified future-cone entropy:

```
H₀ = 0 bits
H₁ = 7 bits
H_n = 12 bits    for n ≥ 2
```

For the uniform byte ensemble on Ω, the uniformization horizon is two steps.

### 8.4 Computational consequence

Repeated uniform byte-averaged mixing beyond this horizon is structurally redundant unless new anisotropy, new gauge bias, or new external coupling is injected. In multi-layer systems, depth is informative only when successive layers add new sector structure instead of reapplying already-saturating local diffusion.

### 8.5 XOR-convolution and spectral composition

Because chirality transport is XOR translation on GF(2)^6, the composition of multiple transport steps is an XOR-convolution. The Walsh-Hadamard transform converts XOR-convolution to pointwise spectral multiplication (Section 5.1.5 of the SDK specification).

For climate transport, this has a direct consequence. If a byte ensemble v is applied repeatedly for n steps, the chirality-space effect after n steps is determined by raising the 64 spectral multipliers to the n-th power:

```
phi_n(s) = phi(s)^n
```

where phi(s) is the single-step spectral multiplier from Section 8.1.

The total cost of computing the n-step climate evolution is one FWHT (to obtain the spectral multipliers), 64 scalar exponentiations, and one inverse FWHT. This cost is independent of n. Dense matrix exponentiation is not required.

For non-stationary ensembles (where the byte bath changes at each step), the composition remains exact: the spectral multipliers at each step are multiplied pointwise. The total cost is O(n x 64 log 64) via repeated FWHT and pointwise multiplication, compared to O(n x 64^2) for dense matrix application.

This spectral composition law is the computational consequence of the kernel's transport being a group action on (GF(2)^6, xor).

---

## 9. Climate Anisotropy

### 9.1 Directional damping

The isotropic ensemble is a special case. In general, the six chirality modes may be driven unequally.

If the q-bits are independent but not equally distributed, with Pr(q_i = 1) = p_i and η_i = 1 − 2p_i, then the exact spectral multiplier factorizes:

```
φ(s) = ∏_{i : s_i = 1} η_i
```

### 9.2 Chirality anisotropy vector

The six per-axis damping parameters define the chirality anisotropy vector:

```
η_vec = (η₁, η₂, η₃, η₄, η₅, η₆)
```

This vector is the complete specification of the anisotropic climate drive.

### 9.3 Diagnostic conditions

**Isotropic climate.** All η_i equal. The shell spectrum is the complete description.

**Directional strain.** Some η_i near ±1 while others are near 0. Specific modes are frozen while others thermalize freely. This is the most common condition in structured computation, where certain operational axes are exercised heavily while others are idle.

**Coherent sector bias.** Low-order anisotropy (small r modes) persists while high-order modes thermalize. The climate carries long-range directional preference.

---

## 10. Gauge Phase Structure

### 10.1 The K4 gauge group

The gate group of the aQPU kernel is the Klein four-group:

```
K4 = {id, S, C, F} ≅ (ℤ/2)²
```

The 2-bit family field of every byte encodes one of four K4 gauge phases. This is part of the state and transport law, not external metadata.

The four gates act on the Gyrostate as:

```
id:  (A, B) → (A, B)
S:   (A, B) → (B, A)
C:   (A, B) → (B ⊕ F, A ⊕ F)
F:   (A, B) → (A ⊕ F, B ⊕ F)
```

where F = 0xFFF.

### 10.2 Character decomposition

Because K4 is abelian with 4 characters α, any observable O on Ω decomposes exactly into gauge sectors:

```
O_α(s) = (1/4) Σ_{g ∈ K4} α(g) · O(g · s)
```

The trivial character sector is the gauge-invariant content. The three nontrivial sectors carry phase-sensitive content.

### 10.3 Gauge transport law

Let μ be a distribution on K4 gauge actions. The exact gauge multiplier on character α is:

```
Γ(α) = Σ_{g ∈ K4} μ(g) · α(g)
```

Each gauge sector evolves independently:

```
G_{t+1}(α) = Γ_t(α) · G_t(α)
```

If the gauge bath is uniform over K4, every nontrivial gauge mode is killed in one step.

### 10.4 Two-bit gauge anisotropy

Because K4 ≅ (ℤ/2)², write gauge coordinates as (a, b). If the two gauge bits flip independently with probabilities p_A and p_B, define:

```
ξ_A = 1 − 2p_A
ξ_B = 1 − 2p_B
```

The character multipliers factorize:

```
Γ_{(a,b)} = ξ_A^a · ξ_B^b    for a, b ∈ {0, 1}
```

The gauge climate has 2 exact damping axes.

---

## 11. The Full Climate Equation

### 11.1 Joint chirality-gauge amplitude

Let C_t(s, α) be the joint chirality-gauge mode amplitude at time t. The full exact update is:

```
C_{t+1}(s, α) = Ψ_t(s, α) · C_t(s, α)
```

with joint multiplier

```
Ψ_t(s, α) = Σ_{q ∈ GF(2)^6} Σ_{g ∈ K4} ν_t(q, g) · (−1)^{⟨s, q⟩} · α(g)
```

for a general joint byte ensemble ν_t on chirality charge and gauge action.

If ν_t factorizes as ν_t(q, g) = ν_χ,t(q) · μ_t(g), then

```
Ψ_t(s, α) = φ_t(s) · Γ_t(α)
```

where

```
φ_t(s) = Σ_q ν_χ,t(q) · (−1)^{⟨s, q⟩}
Γ_t(α) = Σ_g μ_t(g) · α(g)
```

In that factorized case, chirality transport and gauge transport decouple into a tensor product of chirality-spectral and gauge-spectral sectors.

### 11.2 Isotropic reduction

Under isotropic chirality ensemble and factorized gauge ensemble:

```
C_{t+1}(r, a, b) = η_t^r · ξ_A^a · ξ_B^b · C_t(r, a, b)
```

### 11.3 The 8-axis damping system

Combined, the QuBEC occupation dynamics form an 8-axis finite damping system: 6 chirality axes (η₁, …, η₆) and 2 gauge axes (ξ_A, ξ_B). Under isotropic chirality and uniform gauge, this collapses to a 3-parameter system (η, ξ_A, ξ_B) or, with full isotropy, to a single parameter η.

The 8 axes are the total internal degrees of freedom of the climate. Shell-radial content damps as ηʳ. Gauge content damps as ξ_A^a · ξ_B^b. Every climate observable on the QuBEC decomposes into components along these axes.

---

## 12. Exact Partition Algebras

The state partitions of the QuBEC induce exact closed operator algebras. These are finite-dimensional exact quotient classes, not approximations. They provide the structural bridge from climate theory to implementation lowering.
Within each defining symmetry class, these operator classes are closed under linear combination and composition.

### 12.1 Shell-radial operator algebra

The shell-radial operators on GF(2)⁶ form the Bose-Mesner algebra of the Hamming scheme H(6,2). This algebra has dimension 7, one basis element per shell distance class.

The algebra is exactly diagonalized by the Krawtchouk basis. Every shell-radial one-cell operator is represented exactly by 7 spectral coefficients.

### 12.2 Gauge operator algebra

The gauge sector is the group algebra C[K4]. This algebra has dimension 4 and is exactly diagonalized by the four characters of K4.

Every one-cell gauge operator equivariant under K4 action is represented exactly by 4 character multipliers.

### 12.3 Joint shell x gauge algebra

The joint shell-gauge operator class is the tensor product algebra:

```
A_shell ⊗ A_gauge
```

Its dimension is:

```
dim(A_shell ⊗ A_gauge) = 7 · 4 = 28
```

Any shell-radial, gauge-equivariant one-cell operator is therefore an exact 28-parameter operator.

### 12.4 Chirality translation-invariant algebra

Any translation-invariant operator on GF(2)⁶ is diagonal in the 64-point Walsh-Hadamard basis. It is specified exactly by 64 mode multipliers.

### 12.5 Chirality x gauge translation-invariant algebra

Including gauge characters gives the product algebra of chirality translation-invariant and gauge sectors. This exact algebra has dimension:

```
64 · 4 = 256
```

### 12.6 Generic operator and defect decomposition

The one-cell operator hierarchy is:

- shell-radial exact class: 7 parameters
- shell x gauge exact class: 28 parameters
- chirality translation-invariant exact class: 64 parameters
- chirality x gauge translation-invariant exact class: 256 parameters

A generic one-cell operator decomposes into exact quotient-algebra content plus residual defect. The defect is the part not captured by the selected exact partition algebra and is the only component that requires ambient dense treatment.

---

## 13. Arithmetic Substrate

The occupied QuBEC requires an arithmetic engine for evaluating contractions, weightings, and selections over its native charts. The K4 lattice matrix provides this engine through exact decomposition of integer arithmetic into sectors aligned with the manifold structure. This arithmetic layer supplies the concrete realization used by the quotient computations in Section 14.

### 13.1 The dyadic chart

Every signed 32-bit integer v admits a unique decomposition:

```
v = L(v) + B · H(v)
```

where B = 2¹⁶ = 65536, L(v) is the signed low 16-bit value, and H(v) is the signed high 16-bit value.

The low chart L captures the carrier content. The high chart H captures the gauge-scale content. This is a two-digit lattice decomposition with radix 65536, descending from the lattice multiplication method developed independently across multiple mathematical traditions. The abstract mathematical analysis of this decomposition and its relation to common-source factorization, Gram determinants, and chart defects is developed in the companion document (Analysis of Gyroscopic Multiplication, CGM repository).

### 13.2 The K4 lattice matrix

For vectors q, k ∈ ℤⁿ, define the four contraction channels:

```
D₀₀(q, k) = ⟨L_q, L_k⟩       carrier-carrier
D₀₁(q, k) = ⟨L_q, H_k⟩       carrier-gauge (k acts on q)
D₁₀(q, k) = ⟨H_q, L_k⟩       gauge-carrier (q acts on k)
D₁₁(q, k) = ⟨H_q, H_k⟩       gauge-gauge
```

These assemble into a 2 × 2 matrix:

```
M(q, k) = [ D₀₀  D₀₁ ]
           [ D₁₀  D₁₁ ]
```

The ordinary dot product is recovered by the radix projection:

```
⟨q, k⟩ = D₀₀ + B · (D₀₁ + D₁₀) + B² · D₁₁
```

This identity is exact for every int32 dot product. The K4 index set {00, 01, 10, 11} is (ℤ/2)², the same Klein four-group that governs the kernel's gate structure. The arithmetic K4 and the kernel K4 are structurally isomorphic, sharing the (ℤ/2)² organization, but they operate at different layers: one decomposes integer arithmetic, the other governs spinorial gauge phase.

### 13.3 Three operational roles

The four entries of M carry three distinct operational roles.

**D₁₁: chiral alignment.** When H takes values in {−1, 0, +1}, the gauge-gauge contraction is computed by signed support intersection:

```
D₁₁ = popcount(q⁺ ∧ k⁺) + popcount(q⁻ ∧ k⁻)
     − popcount(q⁺ ∧ k⁻) − popcount(q⁻ ∧ k⁺)
```

This counts aligned orientations minus anti-aligned orientations: the same structural role as face alignment under the K4 spinorial group.

**D₀₁ and D₁₀: gauge action on the carrier.** In the spinorial regime, the cross terms act as boolean control masks over the carrier content:

- Where H = +1, L is preserved.
- Where H = −1, L is sign-inverted.
- Where H = 0, L is annihilated.

This is the operational role of a gauge field acting on a state: selective preservation, inversion, or annihilation of carrier content under gauge control.

**D₀₀: carrier contraction.** The contraction of the low charts alone, with no gauge contribution. Pure carrier transport.

### 13.4 Additive sector budget

Define weighted sector magnitudes:

```
E_carrier = |D₀₀|
E_cross   = B · (|D₀₁| + |D₁₀|)
E_align   = B² · |D₁₁|
E_total   = E_carrier + E_cross + E_align
```

The normalized budget:

```
ρ_carrier + ρ_cross + ρ_align = 1
```

describes how the total weighted magnitude distributes across the three operational sectors. This is an exact arithmetic identity.

### 13.5 Three computational regimes

The K4 lattice matrix admits three exact chart regimes determined by the high-chart occupancy.

**Carrier regime.** H_q = 0 and H_k = 0.

```
M(q, k) = [ D₀₀  0 ]
           [ 0    0 ]
```

Only D₀₀ contributes. The sector budget is (1, 0, 0). All gauge-scale content is absent.

**Spinorial regime.** H ∈ {−1, 0, +1}ⁿ for both vectors.

D₁₁ is realized by boolean support intersection (AND + POPCNT on sign masks). D₀₁ and D₁₀ are realized as signed masked actions. All four cells are computed exactly with compressed boolean arithmetic.

**Dense regime.** |H| > 1 at some position.

The same K4 law is evaluated without boolean compression. Correctness is unchanged.

No approximation enters in the regime selection. The three regimes are exact chart specializations of one law. The regime of a pair (q, k) is determined by the high-chart occupancy of the data, not by a precision parameter.

### 13.6 Width-64 structure

Width 64 is the unique arithmetic width at which the following structures coincide:

- the horizon cardinality |H| = 64
- the chirality register GF(2)⁶ = 64 values
- the self-dual mask code C₆₄ = 64 codewords
- the native 64-point Walsh-Hadamard transform dimension
- the 64-byte L1 cache line (6-bit offset)
- the manifold identity |Ω| = 64² = 4096

The GyroLabe tensor surfaces (`gemv64`, `pack_matrix64`) are built around 64-column blocks for this reason. The packed lattice multiplication path operates at the horizon-matched width.

The arithmetic radix satisfies:

```
B = 65536 = |Ω| · 16
```

where the factor 16 is |K4|² = 4². When the radix shifts a digit in the K4 lattice, the shift is exactly one Ω-manifold's worth of states, scaled by the gauge-square factor.

External tensors of arbitrary width d are tiled into ⌈d/64⌉ blocks, each aligned with the horizon-matched payload space. This tiling is not a memory convenience. Each 64-wide block projects onto the 64-element basis generated by the 6 payload bits of the byte. A 768-dimensional transformer axis becomes 12 blocks, a 4096-dimensional axis becomes 64 blocks, each individually addressable by the exact operator classes of Section 12 and the lowering paths of Section 15.

---

## 14. Quotient Computation

This section uses the arithmetic substrate to define exact coarsenings and decision coordinates for one-cell computation.

### 14.1 The natural quotient chain

The architecture exposes exact quotient collapses:

```
256 bytes → 128 Ω-maps → 64 q-classes → 7 shell sectors
```

Each level of the chain discards a specific symmetry (family degeneracy, shadow projection, chirality detail) while preserving the relevant observable content.

In the arithmetic layer, the three computational regimes provide a parallel quotient structure. Carrier data (H = 0) collapses the K4 matrix to a single cell. Spinorial data (H ∈ {−1, 0, +1}) admits boolean compression. Only dense data requires full-precision evaluation.

### 14.2 Sector identification

Many operations that appear as total-order comparisons over a flat candidate space are, within the QuBEC structure, sector-identification problems:

| Task | Flat cost | QuBEC chart | QuBEC cost |
|------|-----------|-------------|------------|
| Nearest mode match | O(V) comparison | q-class chart | O(1) q-map lookup |
| Shell membership | O(V) distance | Shell chart | O(1) popcount |
| Commutation test | 4 kernel steps | q₆ comparison | O(1) equality |
| Radial kernel evaluation | exp + norm | Krawtchouk | 7 polynomial values |
| Normalization | Division | Constant density | Eliminated (d = 0.5 on Ω) |

When the task is sector identification (nearest sector, matching q-class, correct shell, horizon proximity, commutation class), the WHT, q-map, shell structure, and hidden-subgroup machinery provide O(1) or O(log N) identification.

### 14.3 The chart selection principle

When a computation can be performed in multiple exact charts of the same state, the chart in which the operation is structurally regular should be selected.

The QuBEC provides multiple charts in which common operations simplify:

| Operation | Euclidean chart | QuBEC native chart | Simplification |
|-----------|-----------------|-------------------|----------------|
| Normalization | Division by norm | Constant density 0.5 on all of Ω | Eliminated |
| Distance | sqrt of sum of squares | popcount (Hamming) | Integer |
| Radial weighting | exp(−βr²) | λ^N (polynomial) | No transcendental function |
| Mode decomposition | FFT (approximate) | WHT (exact) | Exact finite transform |
| Phase routing | Conditional branch | K4 character projection | Deterministic phase selection |

The principle is not to replace individual functions with faster alternatives. It is to select the chart in which the function becomes structurally trivial.

---

## 15. Practical Lowering to 64-Wide Blocks

This section gives a direct lowering path from learned dense 64-wide blocks to exact one-cell operator classes.

### 15.1 The 64-wide grain

The lowering grain is width 64, using the structural coincidence established in Section 13.6.

### 15.2 Block analysis

Given a learned 64-wide block W, compute:

- shell profile across N = 0,...,6
- gauge character profile on K4 sectors
- 64-point WHT spectral profile
- arithmetic regime profile through K4 lattice decomposition
- residual defect indicators after quotient projections

### 15.3 Exact operator projections

Project W into exact classes in increasing expressivity:

1. shell-radial class (7)
2. shell x gauge class (28)
3. chirality translation-invariant class (64)
4. chirality x gauge translation-invariant class (256)

Each projection is exact inside its class and yields an explicit residual defect.

### 15.4 Runtime execution policy

Execute by class where possible:

- carrier regime path when high charts vanish
- spinorial regime path when high charts are in {−1,0,+1}
- shell-radial exact path for 7-parameter content
- shell x gauge exact path for 28-parameter content
- WHT-diagonal exact path for 64 or 256 multiplier content
- dense residual path only for remaining defect

The structured/residual split provides a natural handoff point for external backends. The exact component is executed natively through GyroLabe. The residual component can be forwarded to any external execution backend (GPU dense path, tensor-network approximation, quantized inference engine) without requiring that backend to adopt the byte formalism. This separation is the operational bridge between the aQPU's exact geometry and external approximate or probabilistic computation. The defect norm (Section 15.3) quantifies what fraction of the workload crosses this boundary.

### 15.5 Routing and depth implications

Routing is phase-sector identification and quotient selection, not branch-heavy control. When ensemble structure is shell or gauge organized, normalization and scoring can be performed in exact quotient coordinates.

Repeated local mixing beyond the uniformization horizon is structurally redundant unless new anisotropy, new gauge bias, or new external coupling is introduced.

---

## 16. Observable Surface

### 16.1 Primary climate observables

| Observable | Definition | Range | Climate content |
|------------|-----------|-------|----------------|
| Occupation density ρ | E(N) / 6 | [0, 1] | Internal excitation fraction |
| Horizon polarization m | 2ρ − 1 | [−1, +1] | Horizon preference |
| Spectral damping η | 1 − 2ρ | [−1, +1] | Higher-mode decay rate |
| Effective support M₂ | 4096 / (1 + η²)⁶ | [64, 4096] | Occupation spread |
| Shell spectrum A(r) | Krawtchouk coefficients | 7 values | Radial harmonic content |
| Chirality anisotropy η_vec | Per-axis damping | 6 values | Directional uniformity |
| Gauge spectrum G(α) | K4 character amplitudes | 4 values | Phase-sector balance |

### 16.2 Secondary observables

Derived from the kernel at any moment:

- horizon_distance
- ab_distance
- shell index
- chirality word χ
- q-class
- parity commitment
- commutativity class
- stabilizer type
- optical coordinates

### 16.3 Empirical estimators from rolling memory

At this point the distinction from Section 1.6 is operational: the formal climate is a measure on Ω and its marginals, while the empirical climate is a finite-window estimator from rolling statistics. The formal quantities connect to implementation through the GyroGraph rolling memories.

**Empirical occupation density.** From `shell_hist7[0..6]` over a window of W observed states:

```
ρ̂ = (1 / 6W) · Σ_{N=0}^{6} N · shell_hist7[N]
```

**Empirical damping.**

```
η̂ = 1 − 2ρ̂
```

**Empirical effective support.** From `chi_hist64[0..63]`:

```
M̂₂ = W² / Σ_{χ=0}^{63} chi_hist64[χ]²
```

This is the Rényi-2 effective support estimated from the chirality histogram.

**Empirical shell spectrum.** The exact Krawtchouk transform of `shell_hist7`:

```
Â(r) = Σ_{N=0}^{6} K_r(N) · shell_hist7[N] / W
```

This is available through `shell_krawtchouk_transform_exact`.

**Empirical chirality spectrum.** The 64-point WHT of `chi_hist64`:

```
spectral64 = wht64(chi_hist64 / W)
```

This is the `spectral64` field of the SLCP record.

**Empirical gauge spectrum.** From `family_hist4[0..3]`, K4 character projection yields 4 gauge-sector coefficients. The trivial character gives the total weight. The three nontrivial characters give the gauge anisotropy.

### 16.4 Implementation mapping

| Formal quantity | Implementation surface | Source |
|-----------------|----------------------|--------|
| ρ | Computed from `shell_hist7` | GyroGraph per-cell |
| η | Derived from ρ | Computed |
| M₂ | Computed from `chi_hist64` | GyroGraph per-cell |
| Shell spectrum | `shell_krawtchouk_transform_exact` | GyroGraph derived |
| Chirality spectrum | `spectral64` via `wht64` | GyroGraph SLCP |
| Gauge spectrum | K4 projection of `family_hist4` | GyroGraph derived |
| Shell transition | Kernel `future_locus_measure` | SDK |
| Thermalization measure | Kernel `future_cone_measure` | SDK |
| Climate transport | Byte bath over `q₆` sectors | GyroLabe `qmap_extract` |

For external systems that produce binary sample traces, trajectory logs, or bitstring outputs, the climate observables provide a projection bridge. Any external binary output stream can be windowed and mapped into chirality histograms, shell marginals, gauge spectra, and effective support estimates using the rolling-memory estimators above. This allows the QuBEC climate framework to serve as a diagnostic and calibration layer over external samplers, simulators, or stochastic hardware without requiring those systems to adopt the byte transition law internally.

All formal quantities of the climate theory have explicit empirical counterparts in the existing stack. The rolling memories (`chi_ring64`, `chi_hist64`, `shell_hist7`, `family_ring64`, `family_hist4`) are the sufficient statistics of the one-cell climate.

---

## 17. Multi-cell Scaling

### 17.1 Multi-cell partition function

For B independent QuBEC cells with identical occupation parameter λ:

```
Z_B(λ) = (64 · (1 + λ)⁶)^B = 64^B · (1 + λ)^{6B}
```

With local occupation parameters λ_c:

```
Z_B({λ_c}) = 64^B · ∏_{c=1}^{B} (1 + λ_c)⁶
```

### 17.2 Total occupation statistics

Let K = Σ_{c=1}^{B} N_c be the total occupation across all cells. For identical cells:

```
K ~ Binomial(6B, ρ)
E(K) = 6Bρ
Var(K) = 6Bρ(1 − ρ)
Var(K / (6B)) = ρ(1 − ρ) / (6B)
```

### 17.3 Concentration

For B = 32 cells (corresponding to a 2048-dimensional space factored into 64-element blocks):

```
Total modes = 192
Var(density) = ρ(1 − ρ) / 192
```

Smooth macroscopic climate variables emerge from exact binary microphysics by concentration. The density variance scales as 1/(6B), giving sharp macroscopic order parameters at moderate cell counts.

### 17.4 Spectral scalability

If the climate kernel on B cells is translation-invariant on GF(2)^{6B}, the exact diagonalizing transform is WHT_{6B}, the Walsh-Hadamard transform on the product space. Large lattices of QuBEC cells remain exactly diagonalizable in the product spectral basis.

---

## 18. Computational Resolution

The climate theory identifies three structural causes of computational cost. Each arises from evaluating an operation in a chart that is not native to the finite quantum medium, and each is resolved by chart selection.

### 18.1 Chart mismatch

**Condition.** Distance, normalization, or weighting is forced into a Euclidean or floating-point chart instead of the machine's native finite chart. In the Euclidean chart, measuring distance requires sqrt, normalizing requires division, and weighting a distribution requires exp.

**Resolution.** On Ω, density is constant at 0.5, eliminating normalization. Distance is a popcount, eliminating sqrt. Weighting is a polynomial in λ (the shell occupation parameter), eliminating exp. The Krawtchouk and Walsh-Hadamard transforms diagonalize all radial and additive processes exactly.

In the arithmetic layer, the K4 lattice matrix provides the same resolution. The three computational regimes (carrier, spinorial, dense) are chart specializations. Carrier data eliminates three of four K4 cells. Spinorial data admits boolean compression. The regime selection is exact and data-determined.

### 18.2 Ensemble mismatch

**Condition.** A population of candidates is treated as a flat list of unrelated scalars instead of as an exactly structured finite ensemble with shells, sectors, and multiplicities.

**Resolution.** The natural quotient chain (256 → 128 → 64 → 7) provides exact algebraic coarsening. Selection tasks that are fundamentally sector-identification problems (nearest q-class, correct shell, commutation class, matching orbit) are resolved by the appropriate quotient, not by total-order comparison across the flat population.

### 18.3 Gauge mismatch

**Condition.** Control and phase are externalized as runtime branching instead of being part of the state and transport law. Decisions that are structurally phase selections are implemented as unpredictable conditional branches.

**Resolution.** The K4 gauge structure carries phase as part of the state. The character decomposition separates gauge-invariant content from phase-sensitive control content deterministically. Routing becomes a character projection, not a conditional jump.

The gauge spectrum from `family_hist4` provides the empirical diagnostic: concentrated gauge spectrum indicates phase-coherent computation; spread gauge spectrum indicates phase-turbulent computation. Within the gauge-character chart, this distinction is decidable without branching.

---

## Repository Context

This document is part of the Gyroscopic ASI Theoretical Formalism and should be read together with:

- Gyroscopic Byte Formalism (byte-level physics, SE(3) generator mapping, palindromic structure, depth-4 closure, SU(2)/SO(3) spinorial structure, cache alignment)
- Gyroscopic ASI aQPU Kernel specification (kernel state, charts, transition law)
- Gyroscopic ASI Quantum Computing SDK specification (QuBEC, Moments, computational spaces, verified quantum advantages, non-Clifford resource)
- Gyroscopic ASI Holographic Algorithm Formalization (boundary-bulk structure, K4 fiber, holographic dictionary, Hodge governance substrate)
- GyroLabe specification (native execution surfaces, packed tensor arithmetic)
- GyroGraph specification (multicellular model, SLCP records, resonance profiles, bridges)
- Physics and aQPU test reports (kernel conformance, Ω topology, quantum structure verification)

The abstract mathematical analysis of multiplication as orthogonal bilinear closure, root extraction, rank-1 common-source factorization, chart defect, and CGM aperture identities is developed separately in Analysis of Gyroscopic Multiplication (CGM repository, DOI: 10.5281/zenodo.17521384).

This document does not depend on the Analysis for implementation or for the validity of its finite thermodynamic results. The Analysis provides the upstream mathematical framework from which the structural parallels between the arithmetic K4, the kernel K4, and the CGM constraint hierarchy derive their interpretation.

The byte is the foundational quantum of action of the aQPU. The climate dynamics formalized here are products of byte transport over the QuBEC manifold. External bridges used in this document enter through 64-wide tensor lowering (Sections 13 and 15) and through empirical climate observables over external traces (Section 16).