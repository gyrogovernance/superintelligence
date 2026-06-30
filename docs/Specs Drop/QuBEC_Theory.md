# QuBEC Theory
## Thermodynamics, Transport, Transform Algebra, and Operator Lowering of the Occupied QuBEC

**Part of the Gyroscopic ASI Theoretical Formalism**

---

## Notation

The following symbols are used throughout Parts I–IV.

| Symbol | Meaning |
|---|---|
| Ω | Reachable manifold, \|Ω\| = 4096 |
| C₆₄ | Self-dual [12,6,2] mask code, 64 codewords |
| GF(2)⁶ | Six-dimensional binary field, chirality register space |
| χ | Chirality word, χ ∈ GF(2)⁶ |
| N | Shell index, N = popcount(χ) ∈ {0,…,6} |
| c | Boundary anchor, c ∈ GF(2)⁶ |
| q | Byte q-charge, q ∈ GF(2)⁶ |
| K4 | Klein four-group of holonomic gates, K4 ≅ GF(2)² |
| η, ηᵢ | Isotropic and per-axis chirality damping parameters |
| ξ_A, ξ_B | Gauge damping parameters on the two K4 axes |
| B | Arithmetic radix, B = 2¹⁶ = 65536 |
| ⊕ | Bitwise XOR |

---

## Overview

This document formalizes the occupied QuBEC: finite quantum thermodynamics, hardware-tier architecture (Part I §6), byte transport and the four-phase processing model (Part II §6a.6), native transform algebra, and operator lowering. The partition function Z₁(λ) = 64·(1+λ)⁶ is exact and polynomial. Spectral transport is diagonal in the Krawtchouk and Walsh-Hadamard bases. Parts: I medium; II transport; III transforms; IV operators; V closure; VI observables; VII verified advantages. Evidence inventory: [hQVM Features Report](../reports/hQVM_Features_Report.md).

Where HQC literature realizes gates through adiabatic or non-adiabatic control loops on quantum hardware, the hQVM instantiates the same geometric structure as a GF(2) finite-state machine on silicon, opening the possibility of structural quantum advantage without quantum hardware.

---

# Part I. Medium & Static Thermodynamics

---

## 1. The Finite Holonomic Medium

### 1.1 The QuBEC manifold Ω

The reachable state space Ω is the set of all 24-bit Gyrostates accessible from the rest state GENE_MAC_REST = 0xAAA555 under the byte transition rule.

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
| Wavefunction | ψ ∈ ℂ^4096 over Ω | Canonical Hilbert lift from [12,6,2] code geometry; eigenspace decomposition of canonical involutions |
| Constitutional | Integer observables | Horizon distance, AB distance, component densities |

Transport observables:

| Chart | Domain | Content |
|-------|--------|---------|
| Gauge and family | K4 phase stream from byte families | Occupied process phase sector |
| q-class | GF(2)⁶ byte charge class | Additive transport sector |
| Parity commitment | Byte parity sector | Constraint and orbit signature |
| Byte sector | 256-byte family and action partition | Transport rule source partition |

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

The shell symmetry |S_N| = |S_{6−N}| and the holographic identity |H|² = |Ω| = 4096 follow from self-duality of C₆₄; the Krawtchouk–MacWilliams connection is developed in §13.4.

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

The broader algebraic structure of the kernel supports Bell-pair factorization over the six modes, CHSH saturation at the Tsirelson bound, Peres-Mermin contextuality, mutually unbiased bases, and a non-Clifford resource. These properties are verified exhaustively in the hQVM test reports. The six modes are the 6 payload bits of the byte, each controlling one oriented dipole pair of the 2x3x2 tensor. They correspond to the 6 generators of se(3): 3 rotational generators from SU(2) in Frame 0 and 3 translational generators from R^3 in Frame 1. Each payload bit executes a discrete pi-rotation around one se(3) basis vector. The shell thermodynamics therefore acts on modes whose quantum structure descends directly from the byte's SE(3) geometry (Gyroscopic Byte Formalism, Section 5.3).

### 1.6 Formal climate object

A one-cell climate at time t is an occupation measure on Ω:

```
p_t : Ω -> [0,1]
Σ_{s ∈ Ω} p_t(s) = 1
```

Stochasticity in the hQVM is exact ensemble structure induced by deterministic byte dynamics: future-cone occupancy measures, shell distributions, q-sector distributions, gauge-sector distributions, and their spectral evolution.

The principal marginals are:

```
π_t(N)      = Σ_{s : N(s)=N} p_t(s)           shell marginal
p_t^χ(χ)    = Σ_{s : χ(s)=χ} p_t(s)           chirality marginal
p_t^g(g)    = process probability of gauge sector g in K4
```

The formal climate is the measure-level object p_t and its quotients. The empirical climate is a finite-window estimator built from rolling histories (`chi_hist64`, `shell_hist7`, `family_hist4`) as empirical summaries of the underlying occupation evolution.

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

From the binomial shell distribution:

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

Because the shell distribution is binomial and the 64-fold holographic degeneracy is uniform, the one-cell Shannon entropy is:

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

Condensation state (M₂, spectral profile, horizon proximity) is the computational medium, not a post-hoc diagnostic. The future cone defines reachable states; current condensation determines whether an incoming byte word deepens coherent occupation or scatters toward thermal equilibrium. Byte-log degeneracy (multiple histories, one state) is condensation capacity, not information loss. Inference is kinematic: the medium gyrates under each 4-byte word; alignment or conflict with current condensation is the sole structural response (extended interpretation: Climate Control Brief).

---

## 6. Hardware-Tier Architecture

### 6.1 Computational levels and kernel realization

The QuBEC architecture maps to the standard CPU memory hierarchy:

| Level | Hardware | Kernel realization | Native operation |
|-------|----------|--------------------|------------------|
| Register | 32-bit CPU register | Register atom: 24-bit state24 + 8-bit intron | Single byte transition |
| L1 Cache | 64-byte cache line, 6-bit offset | Payload space: 6-bit micro-reference to 64 masks | Mask lookup, dipole flip |
| L2 Cache | 64-bit interaction, depth-2 reach | 2-byte composition, 128-state shadow | State reachability |
| L3 / RAM | Shared working memory | Full Ω: 4096 states | State trajectory, climate |
| Disk / Ledger | Persistent append-only storage | Byte log: full provenance | Replay, audit, verification |

### 6.2 Gates as register primitives

The four holonomic gates {id, S, C, F} are register-level primitives: identity, swap, complement-swap, and global complement. Carrier actions and horizon stabilizers are in Part II §10.

### 6.3 Cache alignment

The 6-bit payload and 2-bit family split align with L1 cache addressing (64-byte lines, 6-bit offset). Gyroscopic Byte Formalism §8 develops the correspondence. Width 64 is the structural grain where horizon cardinality, chirality register dimension, mask code size, WHT dimension, and cache line size coincide; external tensors tile into 64-wide blocks (Part IV §19).

### 6.4 Why these numbers

Six modes, four families, 64 masks, 4096 states, 256 bytes, and the 128-state shadow follow from the byte geometry in §1.1 and Kernel specification §2. They are consequences of the CGM constraints on the 24-bit carrier, not independent design choices.

---

# Part II. Byte Transport & Climate Dynamics

---

## 6a. Byte Transport and Excitation Algebra

### 6a.1 Byte charges

Each byte b carries a q-charge:

```
q = q₆(b) ∈ GF(2)⁶
j = popcount(q) ∈ {0, 1, 2, 3, 4, 5, 6}
```

The chirality transport rule is exact:

```
χ' = χ ⊕ q
```

A byte ensemble ν defines a Pauli-diagonal channel on the six-mode register; Walsh eigenvalues φ(s) = Σ_q ν(q)(−1)^{⟨s,q⟩} are the spectral multipliers of §8.1 (WHT forcing: Part III §12.2). The q-weight j = popcount(q) is the number of modes flipped.

### 6a.2 Shell transition rule

At the shell level, a byte with q-weight j maps shell N to shell N' according to:

```
N' = N + j − 2ℓ
```

where ℓ = popcount(χ ∧ q) is the overlap between the current chirality and the byte's q-charge.

For fixed N and fixed j, the exact shell transition probability is:

```
P_j(N → N') = C(N, ℓ) · C(6 − N, j − ℓ) / C(6, j)
```

with ℓ = (N + j − N') / 2. This is an exact hypergeometric transition rule.

### 6a.3 Creation and annihilation at the horizons

From the equality horizon (N = 0): ℓ = 0, so N' = j. A byte of q-weight j creates exactly j excitations from the aligned vacuum.

From the complement horizon (N = 6): ℓ = j, so N' = 6 − j. The same byte annihilates exactly j excitations from the fully occupied pole.

Bytes act as finite creation and annihilation operators on QuBEC shells.

### 6a.4 Byte sectors

The number of bytes with q-weight j is:

```
|B_j| = 4 · C(6, j)
```

The factor 4 is the family degeneracy per q-class. The q-weight distribution of bytes is binomial, with the same combinatorial structure as the shell distribution itself.

### 6a.5 Thermodynamic synthesis

If a byte ensemble is weighted proportionally to λʲ (by q-weight), then from the equality horizon (where j maps exactly to shell j), the induced shell distribution is:

```
π_λ(N) = C(6, N) · λ^N / (1 + λ)⁶
```

The QuBEC occupation law is synthesized by q-weight-biased byte injection from the horizon. Occupation control is steering the distribution by biasing exact byte sectors.

### 6a.6 Processing Model: The Four-Phase Byte Flow

Every byte transition executes a four-phase cycle corresponding to the CGM stage structure. The normative transition rule is Kernel specification §2.6; here we read its CGM phase structure.

**Phase 1: CS (Common Source). Transcription as measurement**

The input byte is transcribed relative to the archetype (`intron = byte ^ GENE_MIC_S`, with `GENE_MIC_S = 0xAA`). This transcription is the measurement of the byte against the common source. Every intron is a mutation measured relative to this single constant. Byte `0xAA` produces intron `0x00` (zero mutation, the identity); all other bytes produce nonzero mutations.

The archetype is a hidden variable: it determines all correlations but cannot itself be observed as a state (Kernel specification §2.2.6). It is the reference frame, not an element of the state space.

**Phase 2: UNA (Unity Non-Absolute). Mutation and asymmetry**

The intron decomposes into a 2-bit family (boundary context) and 6-bit micro-reference (payload). The payload expands to a 12-bit mask through dipole-pair projection (Kernel spec §2.5). Each payload bit controls exactly one oriented dipole pair of the 2×3×2 tensor. The mask mutates the active component A only (`A_mut = A12 ^ mask12`).

Only A is mutated. Component B is not directly affected by the input payload. This asymmetry is the irreducible chirality of the processing pipeline. It is the minimal computational realization of the UNA principle: the active and passive faces are treated differently, introducing variety and preventing absolute unity.

The separation of family (gauge context) from micro-reference (spatial payload) is the discrete analog of separating gauge freedom from physical degrees of freedom in field theory.

**Phase 3: ONA (Opposition Non-Absolute). Gyration and complement-controlled swap**

The mutated active component and the preserved passive component undergo complement-controlled swap. Family bits select whether each component is complemented during the exchange (`invert_a`, `invert_b` from intron bits 0 and 7).

This is the non-associative step (ONA). The gyration mediates the interaction between the mutated active phase (A_mut, carrying the new input) and the preserved passive phase (B12, carrying the historical state).

The family bits select one of four spinorial gauge phases:

- `family = 00`: no inversion, pure swap
- `family = 01`: invert A during swap
- `family = 10`: invert B during swap
- `family = 11`: invert both during swap

These four phases form the Klein four-group K4 = {id, S, C, F}, the gauge structure of the QuBEC (§10).

Opposition is structured: the complement is not negation but chirality inversion. The two horizons (§1.4) are the boundaries where this opposition is fully expressed (complement horizon, A = B^F) or fully collapsed (equality horizon, A = B).

**Phase 4: BU (Balance Universal). Commitment and depth-4 closure**

The new state commits as `state24_next = (A_next << 12) | B_next`. The mutated active content (A_mut) has become the new passive record (B_next). The preserved passive content (B12) has become the new active face (A_next, possibly inverted by family). This role exchange is gyration: temporality as structured rotation, not linear flow.

After four successive byte applications, spinorial closure is achieved:

- Applying any byte four times returns to the starting state: T⁴ = id
- Any alternation of two bytes closes: XYXY = id (verified on Ω)
- Family-phase contributions modulo K4 collapse to net gauge outcome (φ_a, φ_b) ∈ K4

The depth-4 frame is the minimal unit of complete processing. Additional bytes beyond state reachability (depth 2 from rest) contribute provenance (distinct ledger histories reaching the same state), parity structure (trajectory integrity commitments, Kernel spec §3.4), frame records (depth-4 phase organization), and fiber control (K4 gauge phase selection).

**Summary: The byte as fused quantum instruction**

The byte is already a fused quantum instruction packet:

- **Payload** (6 bits, positions 1-6): which of the six dipole modes to mutate (spatial content)
- **Family** (2 bits, positions 0 and 7): which spinorial gauge phase to apply during gyration (chirality context)
- **Provenance atom**: the exact byte value that enters the append-only ledger

The four phases (CS, UNA, ONA, BU) are not external labels. They are the intrinsic temporal structure of the transition rule. Chirality is gyration; gyration is temporality; temporality is the ordered sequence of Moments produced by byte transport.

The 4 holonomic gates {id, S, C, F} are the operations where the mutation step is either trivial (mask = 0) or maximal (mask = 0xFFF) and exactly compensated by the gyration phase (Part I §6.2). They preserve both horizons because they do not create partial chirality. All other bytes create partial transformations that move states between the horizons, populating the contingent bulk of Ω.

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

Code duality and shell transport share this basis (§13.4).

### 7.4 The finite spectral transport rule

If A_t(r) is the amplitude of the r-th Krawtchouk spectral mode at time t, then:

```
A_{t+1}(r) = ηʳ · A_t(r)
```

This is the exact spectral transport rule of the QuBEC. Higher spectral modes (larger r) are damped more rapidly. The rate is controlled by the single parameter η. No differential equation, no discretization error, no truncation: the dynamics is exact and finite.

The mode r = 0 (constant mode) is always preserved: η⁰ = 1. The mode r = 6 (maximum frequency) is damped most aggressively: η⁶. Between these extremes, each mode decays at a rate determined by its shell weight.

This is the same eigenvalue pattern that appears in Pauli-diagonal noise models, where a Pauli-Z string of weight r decays by ηʳ per application. The QuBEC spectral law can therefore serve as an exact finite reference model for channel tomography, randomized benchmarking, and related noise-characterization protocols.

---

## 8. General Climate Transport

### 8.1 Full spectral transport rule

Let ν be an ensemble distribution on q-values in GF(2)⁶. The exact spectral multiplier on Walsh mode s is:

```
φ(s) = Σ_{q ∈ GF(2)⁶} ν(q) · (−1)^{⟨s, q⟩}
```

If A_t(s) is the Walsh amplitude of the climate at mode s, then:

```
A_{t+1}(s) = φ_t(s) · A_t(s)
```

This is the full exact climate transport rule on additive quantum modes. Each Walsh mode evolves independently, with multiplier determined by the Fourier transform of the ensemble distribution.

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

Under the uniform byte ensemble (all 256 bytes equally likely), the one-step shell transition rule collapses to:

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

The total cost of computing the n-step climate evolution is one WHT (to obtain the spectral multipliers), 64 scalar exponentiations, and one inverse WHT. This cost is independent of n. Dense matrix exponentiation is not required.

For non-stationary ensembles (where the byte bath changes at each step), the composition remains exact: the spectral multipliers at each step are multiplied pointwise. The total cost is O(n x 64 log 64) via repeated WHT and pointwise multiplication, compared to O(n x 64^2) for dense matrix application.

This spectral composition identity is the computational consequence of the kernel's transport being a group action on (GF(2)^6, xor).

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

The gate group of the hQVM kernel is the Klein four-group:

```
K4 = {id, S, C, F} ≅ (ℤ/2)²
```

The 2-bit family field of every byte encodes one of four K4 gauge phases. This is part of the state and transport rule, not external metadata.

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

### 10.3 Gauge transport rule

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

### 10.5 Horizon stabilizers and CGM correspondence

| Gate | Complement horizon (S-sector) | Equality horizon (UNA degeneracy) |
|------|-------------------------------|-----------------------------------|
| id | Fixes all 64 pointwise | Fixes all 64 pointwise |
| S | Permutes: 32 two-cycles | Fixes all 64 pointwise |
| C | Fixes all 64 pointwise | Permutes: 32 two-cycles |
| F | Permutes: 32 two-cycles | Permutes: 32 two-cycles |

At the byte level, only {0xD5, 0x2B} fix every complement horizon state pointwise; only {0xAA, 0x54} fix every equality horizon state pointwise.

**CGM reading:** Gate C stabilizes the S-sector (opposition preserves the common source). Gate S stabilizes the equality horizon (non-commutativity is invisible at its own boundary). Gate F stabilizes neither horizon pointwise (balance through dynamics). Gate id is the universal reference.

| CGM Stage | Gate | Fixed-point set |
|-----------|------|-----------------|
| CS | id | All Ω |
| UNA | S | A = B (64 states) |
| ONA | C | A = B ⊕ 0xFFF (64 states) |
| BU | F = S ∘ C | None (depth 2) |

### 10.6 K4 orbit structure on Ω

Under the K4 gate group, Ω partitions into 1056 orbits: 32 of size 2 on the complement horizon (paired by S), 32 of size 2 on the equality horizon (paired by C), and 992 of size 4 in the bulk. Bulk states have trivial K4 stabilizer; horizon states have stabilizer of order 2.

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

# Part III. Native Transform Algebra

Part II derived climate transport in state space. This part names the three exact harmonic bases (Walsh-Hadamard, Krawtchouk, K4 character), states the forcing theorems, records fast execution forms, and classifies exactness. Transport multipliers and damping laws are stated once in Part II §6a-11.

---

## 12. Chirality Spectral Transform

### 12.1 Definition

The chirality spectral transform is the Walsh-Hadamard transform on GF(2)⁶. For any function f : GF(2)⁶ → ℤ:

```
WHT(f)(u) = Σ_{χ ∈ GF(2)⁶} f(χ) · (−1)^{⟨u,χ⟩}
```

where ⟨u,χ⟩ = popcount(u ∧ χ) mod 2. The orthonormal matrix form is:

```
H(u,χ) = (−1)^{popcount(u ∧ χ)} / 8
```

### 12.2 Structural theorem

**Theorem.** The chirality register GF(2)⁶ is an additive group under ⊕. Byte transport on this register is exact XOR translation: χ′ = χ ⊕ q. By Pontryagin duality applied to the finite abelian group (GF(2)⁶, ⊕), the unique Fourier transform of this transport group is the Walsh-Hadamard transform. No other transform diagonalizes XOR-convolution exactly.

**Corollary.** The WHT is not a tool applied to the chirality register. It is the unique exact spectral dual of the transport rule.

### 12.3 Composition rule

Byte transport is XOR-translation (Part II §6a.1). Sequential composition is XOR-convolution on GF(2)⁶, diagonalized by the WHT (Theorem §12.2):

```
(f ∗ g)(χ) = Σ_a f(a) · g(χ ⊕ a)
WHT(f ∗ g)(u) = WHT(f)(u) · WHT(g)(u)
```

Single-step climate update A_{t+1}(u) = φ(u) · A_t(u) with φ from Part II §8.1 is the ensemble special case.

### 12.4 Fast execution form

The 64-point WHT executes via butterfly stages:

```
(x, y) → (x + y, x − y)
```

For n = 64:

- 6 stages
- 32 butterflies per stage
- 192 butterflies total
- 384 additions and subtractions

The 64-point WHT butterfly requires no multiplications. Its execution uses integer additions and subtractions. This statement applies only to the WHT butterfly. It does not apply to Krawtchouk coefficient evaluation, spectral pointwise multiplication, integer contraction, learned-weight application, or general matrix multiplication.

### 12.5 Anisotropic factorization

When the byte ensemble drives the six chirality modes independently, with flip probability pᵢ on axis i, the spectral multiplier factorizes:

```
φ(u) = ∏_{i : uᵢ = 1} ηᵢ,   where ηᵢ = 1 − 2pᵢ
```

The isotropic case ηᵢ = η for all i gives φ(u) = η^{popcount(u)} = ηʳ for mode u of weight r.

---

## 13. Shell Radial Transform

### 13.1 Definition

The shell radial transform acts on the seven shell values N = 0,…,6. The Krawtchouk polynomials on H(6,2) are:

```
K_r(N) = Σ_{j=0}^{r} (−1)^j · C(N,j) · C(6−N, r−j)
```

The shell spectral transform is:

```
A(r) = Σ_{N=0}^{6} K_r(N) · π(N)
```

### 13.2 Structural theorem

**Theorem.** The shell index N = popcount(χ) defines the radial distance classes of the Hamming scheme H(6,2). By the theory of association schemes, the unique family of orthogonal polynomials invariant under all symmetries of H(6,2) is the Krawtchouk family. The Krawtchouk transform is therefore the unique exact radial harmonic basis of the shell quotient.

### 13.3 Radial transport rule

Isotropic shell dynamics diagonalize here as A_{t+1}(r) = Λ(r) · A_t(r) (Part II §7.4, §8.2).

### 13.4 Code-theoretic triple identity

The self-dual [12,6,2] mask code C₆₄ connects three structures through one algebraic fact.

**MacWilliams identity.** For the self-dual code C₆₄, the weight enumerator satisfies:

```
W_C(x,y) = (1/64) · W_C(x+y, x−y)
```

This forces the weight enumerator to be invariant under the Hadamard substitution.

**Krawtchouk diagonalization.** The MacWilliams transform is expressed in the Krawtchouk polynomial basis. The self-duality condition becomes:

```
Σ_i wᵢ · K_r(i) = wᵣ
```

The weight enumerator is an eigenvector of the Krawtchouk transform. This is the code-theoretic expression of the same diagonalization that governs shell transport.

**Horizon self-Fourier property.** The set H_A of A-components of complement-horizon states equals the archetype A12 XOR every element of C₆₄. The Walsh transform of the indicator function of H_A has support exactly on C₆₄ and takes values in {0, 64}.

The Krawtchouk polynomial engine governs code weight distributions, shell radial transport, and horizon boundary structure simultaneously. This is a structural consequence of the self-duality of C₆₄.

### 13.5 Plancherel identity

```
Σ_{χ=0}^{63} |f(χ)|² = (1/64) · Σ_{u=0}^{63} |WHT(f)(u)|²
```

For occupation p(χ), this is the transform-side form of the M₂ condensation identity (§5.2).

---

## 14. Gauge Character Transform

### 14.1 Definition

The gauge transform acts on the four K4 family sectors. For a gauge observable g : K4 → ℝ, the character transform is:

```
G(α) = Σ_{x ∈ K4} α(x) · g(x)
```

where α ranges over the four characters of K4 ≅ GF(2)². In matrix form this is the 4-point Hadamard transform:

```
[[1,  1,  1,  1],
 [1, −1,  1, −1],
 [1,  1, −1, −1],
 [1, −1, −1,  1]]
```

### 14.2 Structural theorem

**Theorem.** The family bits define a gauge sector in K4 ≅ GF(2)². By the character theory of finite abelian groups, the character table of K4 is the unique complete orthogonal decomposition of the gauge sector.

### 14.3 Gauge transport rule

Factorized gauge evolution: G_{t+1}(a,b) = ξ_A^a · ξ_B^b · G_t(a,b) (Part II §10.3–10.4).

### 14.4 Combined chart

Under factorized anisotropy, shell-radial and gauge spectra tensor to the 8-axis damping law of Part II §11.3 (28 modes under shell reduction).

---

## 15. Integer Contractions

For integer vectors q, k ∈ ℤⁿ, contractions use the ordinary dot product ⟨q, k⟩. The dyadic decomposition v = L(v) + B·H(v) (where B = 2¹⁶) is exact bookkeeping. It does not define routing or mandate evaluation paths. Formal dyadic chart structure is developed in *Analysis of Gyroscopic Multiplication*.

### 15.1 Width-64 commensurability

The arithmetic radix satisfies:

```
B = 2¹⁶ = 65536 = |Ω| · 16
```

where 16 = |K4|² = 4². This records structural commensurability between the dyadic radix and the kernel's four-phase gauge organization. It does not establish an arithmetic execution model parallel to kernel K4 gates.

---

## 16. Unified Correspondence

### 16.1 Translation-convolution-spectrum theorem

**Theorem.** The following are equivalent charts of one transport structure:

1. Byte-induced chirality translation: χ′ = χ ⊕ q
2. XOR-convolution on GF(2)⁶: p_{t+1}(χ) = Σ_a ν(a) · p_t(χ ⊕ a)
3. Walsh-Hadamard spectral multiplication: A_{t+1}(u) = φ(u) · A_t(u)

### 16.2 Signed-support correspondence

**Theorem.** When vectors are encoded in the signed-support alphabet {−1, 0, +1}, popcount-based alignment counts and Walsh character correlations measure the same finite geometry under a fixed change of chart coordinates.

In the {−1, +1} encoding where bit 0 maps to +1 and bit 1 maps to −1, the inner product becomes:

```
⟨s_u, s_q⟩ = Σ_i (−1)^{uᵢ ⊕ qᵢ} = 6 − 2 · popcount(u ⊕ q)
```

The signed-support contraction and the Walsh character are dual coordinate expressions of the same correlation structure, related by chart-specific normalization and basis conventions.

### 16.3 Binary and ternary chart complementarity

The transport rule is most natural in the binary chart of GF(2)⁶: state differences, chirality, XOR composition, Walsh characters, and code duality. The contraction law is most natural in the ternary chart {−1, 0, +1}: alignment counting, signed support intersections, and masked inversion or preservation.

Transport is binary and contraction is signed-support. These are two exact alphabets of one medium, each selected by the operation being performed.

### 16.4 Exactness classes

All operations of the native transform algebra fall into two exactness classes.

**Integer exact.**  
Exact over integer arithmetic with no approximation: byte stepping, q-charge extraction, chirality transport, unnormalised WHT, shell counting, parity commitments, and all horizon and shell observables.

**Dyadic exact.**  
Exact as rational values with denominators that are powers of two or small combinatorial factors: normalised WHT, K4 character normalisation, shell probabilities, Krawtchouk-normalised coefficients, and climate occupation fractions. These remain exact and do not require floating-point semantics.

Dyadic exact values are represented as:

```

(numerator: int64, exponent: int8)

```

representing:

```

numerator · 2^{-exponent}

```

Normalisation operations include:

```

WHT normalization:     right shift by 6  (divide by 64)
K4 character norm:     right shift by 2  (divide by 4)

```

When composing dyadic operations, exponents accumulate additively.

**WHT inverse scaling.**  
The 64-point WHT is its own inverse up to factor 64. The 16-point WHT used in the chi-gauge tile is its own inverse up to factor 16. Any implementation dividing by 4096 = 16³ is applying three inverse scalings where only one is required.

---

# Part IV. Execution Consequences

---

## 17. Structured Operators and Native Symmetry Projections

### 17.1 Exact native symmetry classes

The QuBEC admits independent exact operator symmetry classes. An operator may possess zero, one, or multiple native symmetries.

**Shell-radial symmetry.**  
An operator W has shell-radial symmetry when it depends only on the shell index:

```

N = popcount(χ)

```

It is specified by 7 Krawtchouk eigenvalues:

```

λ₀, …, λ₆

```

**Shell × gauge symmetry.**  
An operator W has shell × gauge symmetry when it depends on shell index and K4 gauge sector. This is the tensor product of the shell-radial algebra and the K4 character algebra.

**Chirality translation-invariance.**  
An operator W has chirality translation-invariance when it commutes with all XOR translations on the chirality register. It is diagonal in the Walsh basis and is specified by 64 spectral multipliers:

```

φ(u), u ∈ GF(2)⁶

```

**Chirality × gauge translation-invariance.**  
An operator W has chirality × gauge translation-invariance when it is diagonal in the tensor product of the Walsh basis and the K4 character basis. It is specified by 256 multipliers.

An operator possessing none of these symmetries is unstructured under the native algebra. No additional class is assigned. Its application is evaluated by ordinary dot-product contraction.

### 17.2 Projection and defect

For any operator W and native symmetry class Q, define:

```
P_Q(W) = component of W lying in symmetry class Q
D_Q(W) = W − P_Q(W)
```

The exact operator application is:

```

W · x = P_Q(W) · x + D_Q(W) · x

```

P_Q(W) · x is evaluated through the native transform diagonal corresponding to Q.  
D_Q(W) · x is evaluated as an ordinary dot-product contraction on the stored remainder.

The projection energy ratio is a diagnostic quantity:

```

R_Q(W) = ‖P_Q(W)‖_F / ‖W‖_F

```

where ‖·‖_F is the Frobenius norm.

R_Q(W) measures how much Frobenius energy lies in symmetry class Q. It does not define correctness, routing, class membership, or replacement of the operator application.

---

## 18. Algorithms

### 18.1 n-step chirality evolution

Evolves the chirality climate by n applications of byte ensemble ν without explicit byte replay.

```
Input:  chi_hist64       current climate, int32[64]
        byte_ensemble    distribution ν on q-charges, int32[64]
        n_steps          integer

Output: chi_hist64_next  climate after n steps, int32[64]

1. spectral_climate   ← WHT(chi_hist64)
2. spectral_ensemble  ← WHT(byte_ensemble)
3. spectral_composed  ← spectral_ensemble ^ n_steps     (elementwise power)
4. spectral_result    ← spectral_climate * spectral_composed  (elementwise multiply)
5. chi_hist64_next    ← WHT(spectral_result)
6. chi_hist64_next  >>= 6                               (exact dyadic normalization)

return chi_hist64_next
```

Step 3 uses exponentiation by squaring: cost O(64 · log₂ n) multiplications.

### 18.2 Shell radial evolution

```
Input:  shell_hist7      current shell marginal, int32[7]
        shell_eigenvals  Λ(r) for r = 0..6
        n_steps          integer

Output: shell_hist7_next evolved shell marginal, int32[7]

1. spectral ← Krawtchouk7(shell_hist7)
2. For r = 0..6: spectral[r] ← spectral[r] * shell_eigenvals[r]^n_steps
3. shell_hist7_next ← Krawtchouk7_inverse(spectral)
4. normalize with tracked dyadic denominator

return shell_hist7_next
```

### 18.3 Structure analysis


This procedure computes diagnostic structure measurements for W. It does not define application semantics.

Input:  W, operator matrix of shape (64, 64)
Output: structure_report

1. Compute the translation-invariant projection:
   P_translation ← XOR-circulant projection of W
   R_translation ← ‖P_translation‖_F / ‖W‖_F

2. Compute the shell-radial projection:
   P_shell ← shell-radial projection of W
   R_shell ← ‖P_shell‖_F / ‖W‖_F

3. Compute the shell × gauge projection:
   P_shell_gauge ← shell × K4-character projection of W
   R_shell_gauge ← ‖P_shell_gauge‖_F / ‖W‖_F

4. Compute the chirality × gauge projection:
   P_chi_gauge ← WHT × K4-character diagonal projection of W
   R_chi_gauge ← ‖P_chi_gauge‖_F / ‖W‖_F

5. Return:
   {
   translation_energy_ratio: R_translation,
   shell_energy_ratio: R_shell,
   shell_gauge_energy_ratio: R_shell_gauge,
   chi_gauge_energy_ratio: R_chi_gauge,
   exact_application: "W · x = P_Q(W) · x + D_Q(W) · x"
   }

The returned structure report is used for profiling, inspection, and cost estimation. It is not a semantic branch.

### 18.4 Horizon proximity detection

The **threshold** here is a **spectrum shape** tolerance (how close the empirical WHT coefficients are to binary {0, 64}). It is **not** a projection energy ratio and **not** used for operator or block routing.

```
Input:  chi_hist64   empirical chirality histogram
        threshold    real, default 0.95

Output: is_near_horizon   boolean

1. spectrum ← WHT(chi_hist64)
2. spectrum_normalized ← |spectrum| / max(|spectrum|)
3. For each coefficient:
       rounded ← round(coefficient)
       error   ← |coefficient − rounded|
4. return max(error) < (1 − threshold)
```

Complement-horizon states have Walsh spectra supported on C₆₄ with binary values {0, 64}. The test measures binary concentration of the empirical spectrum.

### 18.5 Anisotropy extraction

```
Input:  byte_ensemble   ν(b) for b = 0..255

Output: eta_vec   per-axis damping, float[6]

For each axis i in {1..6}:
    p_i ← Σ_{b : payload_bit_i(b) = 1} ν(b)
    eta_vec[i] ← 1 − 2 · p_i

return eta_vec
```

---


### 18.6 Symbolic Cost Semantics

The total cost of a climate computation decomposes as:

```
C_total = C_extract + C_transform + C_apply + C_defect + C_memory
```

where:

- C_extract: cost to move the state into the chosen chart
- C_transform: cost of the native transform (WHT, Krawtchouk, K4Char)
- C_apply: cost of diagonal or pointwise application
- C_defect: cost of evaluating the defect **D_Q** by ordinary dot-product contraction
- C_memory: bytes moved and cache effects

### 18.7 Epistemic distinction

Three kinds of cost statements are distinguished.

**Symbolic cost.** Operation-count model on abstract arithmetic in each chart. Exact.

**Architecture-dependent execution cost.** Implementation cost on a concrete pipeline, dependent on memory hierarchy, vector width, and cast paths. Structural estimate.

**Measured benchmark cost.** Wall-time estimate from a target build. Empirical, and requires explicit citation of conditions.

The cost figures given below are symbolic unless stated otherwise.

### 18.8 Native transform costs

| Operation | Arithmetic | Multiplications |
|---|---|---|
| 64-point WHT | 384 add/sub | 0 |
| Krawtchouk7 | 42 multiply-add | 42 |
| K4Char4 | 12 add/sub | 0 |
| WHT + pointwise + inverse WHT | 832 total | 64 |

### 18.9 Multiplication claims

The native transition rule is multiply-free.

The 64-point WHT butterfly is multiply-free.

The K4 character transform is multiply-free.

The Krawtchouk transform uses scalar multiply-add operations.

The spectral application of a diagonalised operator uses pointwise scalar multipliers.

A learned 64 × 64 weight block is multiply-free only when its data and symmetry structure reduce the required contraction to XOR, signed masks, additions, subtractions, and popcount operations.

The specification must not claim that general matrix multiplication is replaced by XOR. The formal claim is that native transport is XOR, native spinorial state projection is multiply-free, and learned weight application is exposed to native structure tests and exact dot-product evaluation where structure does not reduce further.

### 18.10 n-step evolution: native vs dense

**Dense method via matrix exponentiation:**

```
Matrix exponentiation M^n via binary exponentiation:
   approximately log₂(n) matrix multiplies
   Each matrix multiply: 64³ = 262,144 multiply-accumulates

Final matrix-vector product: 64² = 4,096 multiply-accumulates
```

**Native spectral method (§18.1):**

```
2 WHTs:              768 add/sub
n-step power:        64 · log₂(n) multiplies  (exponentiation by squaring)
Pointwise multiply:  64 multiplies
Normalization:       64 right shifts
```

**Symbolic ratio at n = 1000:**

```
Dense:   log₂(1000) · 262,144 ≈ 2,621,440 multiply-accumulates
Native:  768 + 64 · 10 + 64 ≈ 1,472 arithmetic operations

Ratio: approximately 1,800×
```

The symbolic ratio grows with n because both methods use binary exponentiation, but the dense method exponentiates 64×64 matrices while the spectral method exponentiates 64 scalars.

### 18.11 Multi-cell batch

For B independent cells with identical ensemble:

**Dense:**

```
B · 4,096 multiply-accumulates per step
Memory: B · 512 bytes (vectors) + 16,384 bytes (shared matrix)
```

**Batch spectral:**

```
Batch WHT on B cells:    B · 384 add/sub  (parallelizable)
Ensemble WHT (once):     384 add/sub
Pointwise multiply:      B · 64 multiplies
Batch inverse WHT:       B · 384 add/sub
Normalization:           B · 64 right shifts
Memory:                  B · 256 bytes
```

**Symbolic ratio (B = 32):**

```
Dense:   32 · 4,096 = 131,072
Native:  32 · 832 + 384 ≈ 27,008

Ratio: approximately 4.9×
```

The batch ensemble WHT is computed once and broadcast, giving a sublinear cost scaling in B for the ensemble component.

**Concentration.** For B independent cells with identical occupation parameter, the variance of the density estimator scales as:

```
Var(density) = ρ(1 − ρ) / (6B)
```

At B = 32 and ρ = 0.5, the standard deviation of the empirical density estimate is approximately 0.036.

---

## 19. Lowering and Interoperability

### 19.1 The 64-wide lowering grain

Width 64 is the canonical grain at which the following structures coincide:

- horizon cardinality |H| = 64
- chirality register dimension |GF(2)⁶| = 64
- self-dual mask code size |C₆₄| = 64
- 64-point WHT dimension
- 64-byte L1 cache line (6-bit offset)

This is the natural interoperability grain between external tensors of arbitrary width and the native QuBEC computation surfaces. This structural commensurability is established in Part I §6.

### 19.2 Block tiling of external tensors

An external tensor of width d tiles into native blocks as:

```
d = 64k        → k exact blocks
d = 64k + r    → k blocks + 1 residual block, padded to 64
```

Each 64-wide block projects onto the chirality register basis. A transformer layer of width 768 becomes 12 blocks. A width-4096 layer becomes 64 blocks. Each block is individually addressable by the exact operator quotient classes of Section 17.

### 19.3 External tensor ingestion

```
Input:  W       dense weight matrix, shape (rows, cols)

Output: block_registry

1. Pad rows and cols to next multiples of 64 when needed.
2. For each output tile a and input tile b:
       W_block ← W[a·64 : (a+1)·64, b·64 : (b+1)·64]
       decomp  ← decompose64(W_block)
       block_registry[a,b] ← (W_block, decomp)

return block_registry
```

### 19.4 Hybrid block application

Block application is unconditional decomposition application. For each 64x64 block **W_block**:

**apply64(W_block, x_block) = apply_native(P_Q(W_block), x_block) + dot(D_Q(W_block), x_block)** for the selected native class Q, with unconditional decomposition per §17.2.

Registry metadata is cache-only and must not gate this identity.

```
Input:  block_registry, x   input vector of width cols

Output: result vector

1. For each output tile a:
       y_a <- 0
       For each input tile b:
           x_block <- x[b*64 : (b+1)*64]
           (W_block, decomp) <- block_registry[a,b]
           y_a <- y_a + apply64_from_decomp(decomp, x_block)
2. write y_a into result[a*64 : (a+1)*64]
```

A literal **if class ... else matmul** sketch is never the semantic specification of multiplication.

Illustrative transformer bridge examples (KV polar encoding, native attention) are specified in Gyroscopic Runtime Specs Part III.

---

# Part V. Closure Geometry

---

## 20. Aperture Constants and Transform Algebra

### 20.0 Closure notation

| Symbol | Meaning |
|---|---|
| m_a | Observational aperture, m_a = 1/(2√(2π)) |
| δ_BU | BU dual-pole monodromy |
| ρ_cl | Closure ratio, ρ_cl = δ_BU/m_a |
| Δ | Aperture gap, Δ = 1 − ρ_cl |
| Q_G | Quantum gravity invariant, Q_G = 4π |

### 20.1 Aperture constants

The CGM constants governing closure geometry are:

```
m_a    = 1 / (2√(2π))    observational aperture
δ_BU                      BU dual-pole monodromy
ρ_cl   = δ_BU / m_a       closure ratio
Δ      = 1 − ρ_cl         aperture gap  (≈ 0.0207)
Q_G    = 4π               quantum gravity invariant
```

### 20.2 Observational aperture as root extraction

The observational aperture m_a equals the derivative of √x evaluated at x = 2π:

```
d/dx (√x)|_{x=2π} = 1/(2√(2π)) = m_a
```

The central normalization identity is:

```
Q_G · m_a² = 1/2
```

This states that the product of the full solid angle with the squared root-extraction rate at the phase horizon equals the half-integer 1/2, connecting the observational geometry to the SU(2) double-cover structure.

### 20.3 Polar role forced by the aperture

Because Δ > 0, the system is not absolutely closed. A fully closed system (Δ = 0) would permit characterization by a scalar magnitude alone. The irreducible aperture residue Δ > 0 forces the representation to include both the radial coordinate (shell N) and the directional coordinate (chirality χ). Neither is eliminable.

This is the structural reason the canonical decomposition of Ω into (c, χ, N) has three components rather than one. The shell N is the radial magnitude. The chirality χ is the residual orientation. Both are required by the geometry of non-zero aperture.

### 20.4 Depth-4 aperture quantization

The depth-4 closure projects to a 48-bit tensor:

```
4 bytes · 12 bits/byte = 48 bits
48 = 2⁴ · 3   (where 3 is spatial dimensionality)
```

The geometric quantization relation is:

```
48 · Δ ≈ 1
```

Numerically: 48 · 0.0207 = 0.9936, with 0.64% deviation from unity.

The best 8-bit dyadic approximation of Δ is:

```
Q₂₅₆(Δ) = 5/256 ≈ 0.01953
```

expressing the aperture at the byte horizon as 5 ticks open out of 256.

The ratio of the two natural quantization scales is:

```
(1/48) / (1/32) = 2/3
```

where 1/32 corresponds to the turn-normalized monodromy δ_BU/(2π) ≈ 1/32. The factor 2/3 is the ratio of chirality (2 faces, A and B, the spinorial double-cover) to space (3 spatial axes). The aperture exists because mapping a 2-phase chiral spinor onto a 3-axis discrete grid leaves a geometric gap.

### 20.5 Unified defect concept

The transform algebra and the closure geometry share a common structural concept: an exact finite remainder from an ideal closure condition.

**Closure defect.** The aperture gap Δ is the remainder when the BU monodromy falls short of the aperture scale. It is the structural reason the medium has both radial and directional coordinates.

**Anisotropy defect.** When the byte ensemble drives the six chirality axes unequally, the climate departs from the isotropic shell distribution. The anisotropy vector (η₁,…,η₆) measures this defect from the isotropic ideal.

**Quotient defect.** When an operator W does not lie exactly in a quotient class Q, the defect D_Q(W) = W − P_Q(W) measures the departure from exact structured form. **D_Q(W) · x** is evaluated as an ordinary dot-product contraction on the stored remainder. Optional ambient matmul is an external offload, not the theory's definition of the defect application.

All three are structurally the same kind of object: a finite remainder from an exact closure ideal, quantifiable, and segregable from the exact part.

### 20.6 Toroidal gate homology

The gate group K4 = (ℤ/2)² is the first homology group of the torus with ℤ/2 coefficients: H₁(T², ℤ/2) = (ℤ/2)². Meridional cycle ↔ S (non-commutativity); longitudinal cycle ↔ C (non-associativity); diagonal cycle ↔ F = S ∘ C (balance); trivial cycle ↔ id. Residual holonomy after both fundamental cycles is δ_BU (§20.1).

---

# Part VI. Observable Surface & Multi-Cell Scaling

---

## 21. Observable Surface

### 21.1 Climate observable surface

Order parameters ρ, m, η, and M₂ are defined in §3. Additional surface quantities:

| Observable | Definition | Range |
|------------|-----------|-------|
| Shell spectrum A(r) | Krawtchouk coefficients | 7 values |
| Chirality anisotropy η_vec | Per-axis damping (§9.2) | 6 values |
| Gauge spectrum G(α) | K4 character amplitudes | 4 values |

### 21.2 Secondary observables

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

### 21.3 Empirical estimators from rolling memory

At this point the distinction from Section 1.6 is operational: the formal climate is a measure on Ω and its marginals, while the empirical climate is a finite-window estimator derived from rolling histories. The formal quantities connect to implementation through the runtime rolling memories.

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

**Empirical shell spectrum.** The exact Krawtchouk transform applied to a rolling shell histogram:

```
Â(r) = Σ_{N=0}^{6} K_r(N) · shell_hist7[N] / W
```

This is available through `shell_krawtchouk_transform_exact`.

**Empirical chirality spectrum.** The 64-point WHT of rolling chirality counts:

```
spectral64 = wht64(chi_hist64 / W)
```

This is the `spectral64` field of the SLCP record.

**Empirical gauge spectrum.** From `family_hist4[0..3]`, K4 character projection yields 4 gauge-sector coefficients. The trivial character gives the total weight. The three nontrivial characters give the gauge anisotropy.

### 21.4 Implementation mapping

| Formal quantity | Implementation surface | Source |
|-----------------|----------------------|--------|
| ρ | Computed from `shell_hist7` | Runtime per-cell |
| η | Derived from ρ | Computed |
| M₂ | Computed from `chi_hist64` | Runtime per-cell |
| Shell spectrum | `shell_krawtchouk_transform_exact` | Runtime derived |
| Chirality spectrum | `spectral64` via `wht64` | Runtime SLCP |
| Gauge spectrum | K4 projection of `family_hist4` | Runtime derived |
| Shell transition | Kernel `future_locus_measure` | SDK |
| Thermalization measure | Kernel `future_cone_measure` | SDK |
| Climate transport | Byte bath over `q₆` sectors | SDK `qmap_extract` |

For external systems that produce binary sample traces, trajectory logs, or bitstring outputs, the climate observables provide a projection bridge. Any external binary output stream can be windowed and mapped into chirality histograms, shell marginals, gauge spectra, and effective support estimates using the rolling-memory estimators above. This allows the QuBEC climate framework to serve as a diagnostic and calibration layer over external samplers, simulators, or stochastic hardware without requiring those systems to adopt the byte transition rule internally.

All formal quantities of the climate theory have explicit empirical counterparts in the existing stack. The rolling memories (`chi_ring64`, `chi_hist64`, `shell_hist7`, `family_ring64`, `family_hist4`) are the empirical summaries used to estimate one-cell climate observables.

---

## 22. Multi-cell Scaling

### 22.1 Multi-cell partition function

For B independent QuBEC cells with identical occupation parameter λ:

```
Z_B(λ) = (64 · (1 + λ)⁶)^B = 64^B · (1 + λ)^{6B}
```

With local occupation parameters λ_c:

```
Z_B({λ_c}) = 64^B · ∏_{c=1}^{B} (1 + λ_c)⁶
```

### 22.2 Total occupation statistics

Let K = Σ_{c=1}^{B} N_c be the total occupation across all cells. For identical cells:

```
K ~ Binomial(6B, ρ)
E(K) = 6Bρ
Var(K) = 6Bρ(1 − ρ)
Var(K / (6B)) = ρ(1 − ρ) / (6B)
```

### 22.3 Concentration

For B = 32 cells (corresponding to a 2048-dimensional space factored into 64-element blocks):

```
Total modes = 192
Var(density) = ρ(1 − ρ) / 192
```

Smooth macroscopic climate variables emerge from exact binary microphysics by concentration. The density variance scales as 1/(6B), giving sharp macroscopic order parameters at moderate cell counts.

### 22.4 Spectral scalability

If the climate kernel on B cells is translation-invariant on GF(2)^{6B}, the exact diagonalizing transform is WHT_{6B}, the Walsh-Hadamard transform on the product space. Large lattices of QuBEC cells remain exactly diagonalizable in the product spectral basis.

---

## 23. Computational Resolution

The climate theory identifies three structural causes of computational cost. Each arises from evaluating an operation in a chart that is not native to the finite holonomic medium, and each is resolved by chart selection.

### 23.1 Chart mismatch

**Condition.** Distance, normalization, or weighting is forced into a Euclidean or floating-point chart instead of the machine's native finite chart. In the Euclidean chart, measuring distance requires sqrt, normalizing requires division, and weighting a distribution requires exp.

**Resolution.** On Ω, density is constant at 0.5, eliminating normalization. Distance is a popcount, eliminating sqrt. Weighting is a polynomial in λ (the shell occupation parameter), eliminating exp. The Krawtchouk and Walsh-Hadamard transforms diagonalize all radial and additive processes exactly.

### 23.2 Ensemble mismatch

**Condition.** A population of candidates is treated as a flat list of unrelated scalars instead of as an exactly structured finite ensemble with shells, sectors, and multiplicities.

**Resolution.** The natural quotient chain (256 → 128 → 64 → 7) provides exact algebraic coarsening. Selection tasks that are fundamentally sector-identification problems (nearest q-class, matching shell, commutation class, matching orbit) are resolved by the appropriate quotient, not by total-order comparison across the flat population.

### 23.3 Gauge mismatch

**Condition.** Control and phase are externalized as runtime branching instead of being part of the state and transport rule. Decisions that are structurally phase selections are implemented as unpredictable conditional branches.

**Resolution.** The K4 gauge structure carries phase as part of the state. The character decomposition separates gauge-invariant content from phase-sensitive control content deterministically. Routing becomes a character projection, not a conditional jump.

The gauge spectrum from `family_hist4` provides the empirical diagnostic: concentrated gauge spectrum indicates phase-coherent computation; spread gauge spectrum indicates phase-turbulent computation. Within the gauge-character chart, this distinction is decidable without branching.

---


---

# Part VII. Verified Computational Advantages

## 24. Verified Computational Advantages

The following advantages are structural invariants of the hQVM, verified by exhaustive testing with exact integer arithmetic.

### 24.1 Hidden subgroup resolution

The q-map q6: {0,…,255} → GF(2)⁶ is a native hidden subgroup structure with uniform 4-to-1 fibers. The Walsh-Hadamard transform on the chirality register resolves the subgroup in 1 step. Classical worst case: O(64) queries.

### 24.2 Deutsch-Jozsa discrimination

The WHT on the chirality register distinguishes constant from balanced functions with probability 1 in 1 step. Classical worst case: 33 queries.

### 24.3 Bernstein-Vazirani secret recovery

The WHT recovers any 6-bit secret string with probability 1 in 1 step. Classical requirement: 6 queries.

### 24.4 Exact two-step uniformization

For any source state in Ω, all 256² length-2 byte words produce exact uniform occupancy over Ω. Each of the 4096 states is reached exactly 16 times. Classical random walk mixing on a 4096-state graph: O(log 4096) ≈ 12 steps.

### 24.5 Holographic compression

The holographic identity |H|² = |Ω| enables encoding any Ω state with log₂(64) + log₂(4) = 8 bits instead of log₂(4096) = 12 bits. This is 33.3% structural compression.

### 24.6 Exact commutativity decision

Whether two bytes commute is determined in O(1) by comparing q6(x) and q6(y). Classical verification requires 4 kernel steps.

### 24.7 Exact tamper detection

Every tamper miss has an exact algebraic explanation:

- Substitution: detected unless replacement is the shadow partner (miss rate 1/255)
- Adjacent swap: detected unless q(x) = q(y) (miss rate ~3/255)
- Deletion: detected unless the deleted byte is a gate stabilizer of the prefix state (bulk states: never)

### 24.8 Householder structure and Grover-speedup potential

A Householder transformation is a reflection about a hyperplane with unit normal vector v, defined as H_v(x) = x − 2⟨x, v⟩v. The Householder matrix P = I − 2vv* is Hermitian, unitary, and involutory (P² = I), with eigenvalues +1 (multiplicity n−1) and −1 (multiplicity 1), and det(P) = −1.

Gate F on Ω acts as (u, v) → (u ⊕ 63, v ⊕ 63). This is an involution with +1 and −1 eigenspaces of equal dimension 2048 and no fixed points, structurally identical to a Householder reflection on the 4096-state manifold. The byte-level fold disagreement (the Z2 curvature seed at the BU boundary) propagates through depth-4 spinorial closure to produce this carrier-level Householder involution.

In gate-model quantum computing, Grover's algorithm rests on two Householder reflections: the oracle U_ω and the diffusion operator U_s = 2|s⟩⟨s| − I. Their composition U_s U_ω is a net rotation in the two-dimensional subspace spanned by |s⟩ and |ω⟩, giving quadratic speedup. The hQVM realizes this geometric structure natively: each K4 gate is a Householder involution on Ω, and the composition of two such involutions produces a net rotation on the carrier state space. This opens the possibility of Grover-type quadratic speedup using exact integer arithmetic on standard silicon, without quantum hardware.

---

## Repository Context

Read with [Gyroscopic_ASI_Specs.md](../Gyroscopic_ASI_Specs.md), [Gyroscopic_ASI_SDK_Quantum_Computing.md](../Gyroscopic_ASI_SDK_Quantum_Computing.md), and [Gyroscopic_ASI_Runtime_Specs.md](../Gyroscopic_ASI_Runtime_Specs.md).
Optional: [Gyroscopic_ASI_Specs_Formalism.md](Gyroscopic_ASI_Specs_Formalism.md), [Gyroscopic_ASI_Holography.md](Gyroscopic_ASI_Holography.md), [Analysis_Gyroscopic_Multiplication.md](../references/Analysis_Gyroscopic_Multiplication.md).
