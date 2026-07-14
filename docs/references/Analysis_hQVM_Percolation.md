# Analysis: hQVM Percolation

## Percolation in the Common Governance Model: The Universality Rule of Ancestry Preservation

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**Reproducibility:** `experiments/hqvm_percolation_analysis_results.txt` and `experiments/hqvm_percolation_analysis_5_results.txt`. Scripts and protocol: Appendix B. Cross-references to `docs/Findings/Analysis_Gravity.md` denote the full findings manuscript (Sections 1 through 8 and Appendices A through H), not the shorter Gravity Note unless explicitly cited.

**Subject classes (arXiv-style):** math-ph; cs.LG; cs.IT; math.PR; math.CO; cs.AI

**Keywords:** Common Governance Model, generator-restricted percolation, GF(2) transport rank, finite transformation semigroups, exact enumeration benchmark, matroid rank, coding theory, mechanistic interpretability, representation learning ground truth, mathematical physics

**Scope and intended reader.** This document assumes familiarity with finite algebra, coding theory, and percolation on random subgraphs, and treats the CGM axiomatic kernel as a first-class mathematical object rather than illustrative metaphor. The percolation census and square-root identities are verified by exhaustive enumeration on the fixed 4096-state kernel. Readers interested in benchmark design for mechanistic interpretability may begin with Section 1.8 and Appendix A. Audience, cross-references, and extended reading paths are collected in `docs/Findings/Analysis_hQVM_Percolation_Note.md`.

## Abstract

The Common Governance Model (CGM) is an axiomatic framework whose foundational requirement is ancestry preservation, meaning that every distinguishable state must remain traceable through recursive operations. That requirement couples two demands that the axioms treat as inseparable. The reference frame at each transition must remain recoverable, which is the condition called identity in the CGM formalism. Transitions must also yield distinguishable outcomes, which is the condition called individuality. The carrier architecture that satisfies both demands splits into two conjugate faces and closes them through depth-four composition. This necessity fixes a specific computational architecture, a 24-bit carrier updated by 8-bit byte operators that act as global permutations on a 4096-state reachable set Omega.

In this architecture every observable cluster is the square of a transport root on the conjugate product U x V, so the exponent is fixed by the product geometry before any random restriction is applied. Restricting the byte alphabet degrades the transport rank on the chirality register, and the reachable cluster shrinks as the square of the surviving root dimension.

We probe this structure with generator-restricted percolation. The percolation parameter p is the independent probability that each of the 256 byte operators is included in the allowed set A. Because the carrier splits into two conjugate faces, the reachable set factorizes as

```
Omega = U x V,   |U| = |V| = 64,   |Omega| = |H|^2 = 4096
```

Here U and V are the two 64-element marginal factors of the product, H is the constitutional horizon (the boundary root of cardinality 64), and |Omega| is the reachable state count. Under fiber-complete restriction the Square-Root Cluster Theorem states that the reachable set from rest has marginal factors U_R and V_R with |U_R| = |V_R| = root(A), and for r(A) at least 1,

```
|Reach(A)| = root(A)^2 = (2^r(A))^2
```

In log2 coordinates this is the linear identity log2|Reach(A)| = 2 r(A), with slope 2 set by the product geometry. The identity holds at every transport rank under fiber-complete restriction and across the hQVM(d) kernel family, where chirality dimension d generalizes the physical instance d = 6 studied here. For one restriction protocol, generator inclusion by micro-reference payload, the full percolation event admits an exact closed-form probability distribution, derived from the coset structure of the byte's family gauge, and the resulting threshold converges to a finite constant as d grows. Five separable coverage thresholds turn on at distinct generator fractions p as inclusion depth increases, from full transport rank through uniform defect coverage to depth-four holonomy closure.

The percolation census reports how plaquette normalization D(A), holonomy coverage micro_cov(p), and shell connectivity C(k) saturate to kernel limits at the complete alphabet and degrade under partial generator access. Principal limits are D = 24, tau_cycle/Delta = 7591/7392, and the full-alphabet inward connectivity profile C(k). Appendix A specifies supervised benchmark tasks on this verified generative structure for studies in representation learning and mechanistic interpretability.

## 1. Introduction

### 1.1 Percolation Theory and Its Standard Setting

Percolation theory describes the emergence of large-scale connectivity in systems where individual connections are opened or closed at random. In the classical formulation, a lattice or graph is given, each bond is declared open independently with probability p, and the central question is whether an open path spans the system. The critical probability p_c marks a geometric phase transition. Below p_c, only finite clusters exist. Above p_c, a spanning cluster appears. The framework models large-scale connectivity in systems from porous media to epidemic spreading by treating individual connections as random variables (Broadbent and Hammersley, 1957).

The standard formulation assumes a pre-existing graph whose connectivity is otherwise undetermined, so randomness is the only tool available to discover its large-scale structure. Universality in this setting means that different microscopic models share the same macroscopic critical behavior near p_c because a small set of invariants controls connectivity at the scaling limit. Correlation length diverges, and critical exponents classify the approach to p_c on families of graphs whose size grows without bound.

The present study operates on a fixed finite kernel with 4096 states. Byte-fraction and structural coverage thresholds are 50% onset points estimated by Monte Carlo with n = 150 through 300 samples (Appendix B). Register-protocol thresholds for Q6-class and micro-reference restriction are exact rank thresholds from closed-form GF(2) rank distributions. Binary matroid structure has appeared as an organizing principle for percolation through blocking clutters (McDiarmid, 1981). The hQVM kernel is a finite realization in which transport rank, cluster size, and coverage observables close exactly under fiber-complete restriction. The universality object of this document is a functional identity on cluster size at every transport rank, together with the restriction scalars pi_j(A), D(A), and Delta that track how open each transport channel remains as p varies.

### 1.2 Generator Restriction, Porosity, and the Functional Universality Identity

Ancestry preservation forces the state space to factorize as a product of two conjugate faces. An observable cluster is therefore always the square of a root process on each face. In log2 coordinates, which decompose a product into additive contributions from equal conjugate factors, squaring becomes a linear relation with slope 2:

```
log2|Reach(A)| = 2 r(A)     for r(A) >= 1 under fiber-complete restriction
```

r(A) is the GF(2) transport rank of the allowed byte set A. The slope 2 is the exponent from squaring each conjugate factor, as in log2|Omega| = 2 log2|H|. The square-root cluster relation is the percolation expression of that product geometry.

**Generator restriction.** The percolation parameter p is the independent probability that each byte is included in the allowed set A. This is bond occupancy on the 256-generator graph: included bytes are open transitions, excluded bytes are closed. Coverage events E_span, E_full, and the coverage hierarchy below are Boolean percolation transitions on the subgraph induced by A.

**Porosity measures.** In porous-media percolation, porosity is the fraction of open pore volume; transport appears only above a critical porosity where a spanning cluster forms (Broadbent and Hammersley, 1957). On the hQVM kernel, restriction is tracked by boundary-channel porosity pi_j(A), plaquette defect census D(A), and aperture gap Delta. These are three coordinates of the same spinorial closure geometry: boundary channels record which constitutional faces remain open under A, plaquette defects record commutation curvature in GF(2)^d, and Delta is the depth-4 residual after fold averaging on Omega. As p and |A| vary, percolation thresholds move all three together through the shared restriction of the byte alphabet.

**The hQVM(d) family.** The CGM axioms fix n = 3 spatial dimensions and six operational degrees of freedom, which select chirality dimension d = 6 for the executable kernel studied here. The algebraic structure generalizes to arbitrary d before that selection. For each d, define chirality register GF(2)^d, state space Omega_d = U_d x V_d with |U_d| = |V_d| = 2^d, horizon |H_d| = 2^d, and byte alphabet of size 2^{d+2} (four family variants per micro-reference). The spinorial transition rule is unchanged. Under fiber-complete restriction the square-root identity holds for every d:

```
|Reach_d(A)| = (2^r(A))^2,    log2|Reach_d(A)| = 2 r(A)
```

Two distinct restriction protocols act on the transport register GF(2)^d. Under Q6-class restriction, each of the 2^d transport values is included independently with probability p, and full rank is the direct spanning event on GF(2)^d. Under micro-reference restriction, each of the 2^d payloads is included independently with probability p; each included payload contributes two transport values through the family action (Section 2.5). The two protocols have distinct closed-form thresholds (Section 4.3.5). At d = 6, the exact rank thresholds are p_c approximately 0.1053 (Q6-class) and p_c approximately 0.0908 (micro-reference).

| d | \|Omega_d\| | \|A_d\| |
|---|-------------|---------|
| 1 | 4 | 8 |
| 2 | 16 | 16 |
| 3 | 64 | 32 |
| 4 | 256 | 64 |
| 5 | 1024 | 128 |
| 6 | 4096 | 256 |
| 7 | 16384 | 512 |
| 8 | 65536 | 1024 |

The hQVM(d) family. d = 6 is the physical chirality dimension of the executable kernel; other values of d verify structural identities and carry exact register-protocol thresholds for d = 4 through 8 (Section 4.3.5).

### 1.3 The Square Root in Standard Percolation

Square roots enter standard percolation only as emergent statistics. The root-mean-square displacement of a random walker in a percolation cluster, sqrt(<r^2(t)>), measures anomalous diffusion at criticality. Exact critical exponents in dependent percolation models carry explicit algebraic square roots. Corner percolation on Z^2 (Toth) yields exponents governed by sqrt(17):

```
gamma = (5 - sqrt(17)) / 4
delta = (sqrt(17) + 1) / 4
```

In this corner-percolation model, gamma and delta are critical exponents governing cluster size and correlation length at the scaling limit, derived from a singular sixth-order ordinary differential equation.

In every case the square root appears at the level of measured output. The nodes themselves are treated as fundamental scalars, and connectivity is built by additive path formation. The present study instead examines a product state space Omega in which the square root is an algebraic property of the reachable set itself. Ancestry preservation forces a conjugate product structure, so the cluster size is a perfect square, and the root of the cluster is the surviving transport dimension.

### 1.4 Ancestry Preservation and the Spinorial Root

The Common Governance Model (CGM) is an axiomatic framework for mathematical physics. It begins from a single operational requirement for ancestry preservation, which demands that every distinguishable state remain traceable through recursive operations. In the CGM axioms, identity names the requirement that the reference frame remain recoverable across transitions, and individuality names the requirement that transitions produce distinguishable outcomes. The carrier satisfies both only when it splits into two conjugate faces whose product forms the observable space.

In relativistic quantum mechanics, the observable mass of a particle is the square root of the squared energy-momentum relation. Dirac took this square root as a first-order operator and recovered the spinor, an object that requires a 720-degree rotation to return to its origin. The square root of a squared invariant is a directed, chiral half-cycle, while the squared invariant is the observable quantity and its root is the spinorial process that generates it. This parallel is recorded as structural motivation for the square-root framing.

In gyrogroup algebra, unique 2-divisibility defines the square root and supplies geodesic midpoints in non-associative, curved spaces (Ungar). The square root stabilizes composition where associativity fails. The CGM transition rule is a discrete gyrogroup composition in which a single byte acts as a gyration, a spinorial half-cycle that inverts chirality without returning to the rest reference until a depth-four canonical word composes two gyrations and squares the half-cycle to close an invariant loop that preserves ancestry.

The state space Omega is the square of this root. Its 4096 states factorize as

```
Omega = U x V,   |U| = |V| = 64,   |Omega| = |H|^2 = 4096
```

The factor H is the constitutional horizon, the 64-element boundary set that must be squared to produce the full space. U and V are its two conjugate marginal realizations on the active and passive gyrophases. Percolation under alphabet restriction measures how much of this root remains accessible.

### 1.5 The hQVM Kernel

The executable kernel studied here is the Holonomic Quantum Virtual Machine (hQVM). Its reachable state set Omega contains 4096 carrier states. Connectivity is fixed by the CGM transition rule. Kernel invariants of the state space are verified on all 4096 states in `docs/reports/hQVM_Features_Report.md`.

The hQVM is the minimal finite realization of the CGM conditions: a two-face 24-bit carrier updated by byte operators on a 256-symbol alphabet. Section 2 gives the state-space layout, byte classification, and canonical word operators.

The alphabet is the product K4 x GF(2)^6: four family phases over 64 transport classes. Six binary operational modes force 2^6 = 64 masks on the twelve-bit faces; oriented dipole pairs realize one mode per pair-flip. An eight-bit label is the fixed-width binary index log2(256). Plaquette total D = 24 is a census theorem at the complete alphabet.

### 1.6 Generator-Restricted Percolation

This study defines percolation on the generator semigroup of byte transition operators. The native percolation parameter is p, the independent inclusion probability of each byte in the allowed set A. Opening a byte includes its global transition on Omega. Closing a byte excludes that transition entirely.

Random restriction here acts on generators for a finite transformation semigroup on Omega, and threshold phenomena are transitions in semigroup generation.

The allowed set restricts generators of a permutation semigroup on Omega. In random generation of finite groups, a small random set of permutations often generates the full symmetric group (Dixon, 1969). The CGM byte set is structured by four family phases and 64 chirality transport classes. Reaching all 4096 states requires diversity across transport classes, and the precise condition is the rank condition r(A) = 6 on the transport register.

Two percolation events are tracked throughout, namely horizon spanning and full reachability from rest. Horizon spanning requires a path from the rest anchor on one horizon to the opposite horizon, while full reachability requires the reachable set to equal all 4096 states.

Generator restriction appears in two natural forms throughout this study. Restriction by transport value acts directly on the chirality register: each of the 2^d transport classes is included as a unit. Restriction by payload acts on micro-reference groups whose transport images are paired under the family action. Both forms are used below; the exact rank thresholds differ (Section 4.3.5).

### 1.7 The Universal Scaling Rule

The scaling parameter in this study is generator restriction. The state space Omega is fixed at 4096 states. The analysis varies the allowed generator set A by independent inclusion with probability p, or by inclusion of structured generator groups such as families, transport classes, or micro-references. The primary observables are reachability from the rest anchor, shell support, and transport statistics derived from the chirality transport values q6(b).

The universal scaling rule is the Square-Root Cluster Theorem. Under fiber-complete restriction, the reachable set from rest has marginal factors U_R and V_R with |U_R| = |V_R| = root(A), where

```
root(A) = 2^r(A)   for r(A) >= 1
root(A) = 2        for r(A) = 0
```

The root dimension root(A) is the cardinality of the surviving transport subspace on each marginal factor before squaring.

```
|Reach(A)| = root(A)^2 = (2^r(A))^2
```

|Reach(A)| counts states reachable from rest under A. U_R and V_R are the reached subsets of the marginal factors U and V. At r(A) = 0, root(A) = 2 and |Reach(A)| = 2. Full connectivity requires root(A) = |H| = 64, equivalently r(A) = 6. Restricting the alphabet degrades the root multiplicatively, and the observable cluster shrinks as the square of that degradation for r(A) at least 1.

The scaling rule has five consequences established in this study.

**Large generator-set saturation.** Random byte subsets of growing size drive percolation observables to fixed kernel values under byte sweeps and structural observables.

**Fiber-complete product clusters.** Under fiber-complete restriction, reachable sets from rest are product clusters governed by transport rank.

**Coupon-collector scaling.** Grouped restriction protocols follow independent-group sampling baselines with distinct thresholds across family, Q6-class, and permutation-class protocols.

**Exact thermodynamics under micro-reference restriction.** For the micro-reference protocol, every coverage rank has a closed-form probability, yielding an exact equation of state for the reachable fraction and its susceptibility, and an asymptotic threshold law as d grows.

**Hierarchy of coverage thresholds.** Orbit reachability, defect-spectrum completion, channel isotropy, two-step uniformization, and holonomy transport turn on at separable generator fractions p as each coverage criterion requires more included bytes or deeper word closure (Section 5). The layered onsets provide a controllable hierarchy of coverage criteria on identical generator subsets, useful when testing which level of structure a probe or learned model encodes.

**Kernel census saturation.** Plaquette normalization D(A) and holonomy transport reach census limits D = 24 and tau_cycle/Delta = 7591/7392 at the complete alphabet (Sections 5.4 and 6.6).

**Scope.** This study reports generator-restricted percolation on the fixed 4096-state CGM kernel and the saturation limits of restricted observables at the complete alphabet.

### 1.8 A Verified Ground-Truth System for Representation Studies

Beyond its role in the CGM programme, this system supports supervised benchmark design for representation learning and mechanistic interpretability. Appendix A specifies the task families; the percolation census in the main text supplies the verified generative structure those tasks assume.

Research on learned representations relies on synthetic datasets whose generative structure is known in full, so a probing method can be scored against enumerated ground truth. Statistical percolation at criticality carries a familiar difficulty: the target structure is subtle, models can fit spurious correlates, and verifying which correlate a probe has detected is itself hard. This kernel avoids that ambiguity for its own labels because every reachable set, shell profile, and coverage flag is computed by enumeration on 4096 states.

The testbed has four structural properties relevant to such evaluations. First, labels are exact by census, not by sampling. Second, the hidden mechanistic variable is GF(2) transport rank, with a closed-form map to cluster size under fiber-complete restriction, derived from the axiomatic kernel verified across tiered feature gates (`docs/reports/hQVM_Features_Report.md`). Third, the latent computation follows a fixed chain: allowed bytes determine transport values, their span defines rank r(A), and |Reach(A)| = (2^r(A))^2. A model trained on membership vectors to predict cluster size could satisfy the labels by counting active generators or by reconstructing the span; Appendix A.1 separates those routes with size-controlled strata. Fourth, a built-in dynamics shuffle preserves marginal label statistics while replacing the label-to-dynamics map (Appendix A.2), giving a matched-pair control for mechanism-versus-correlate studies. Training models, running probes, and reporting interpretability outcomes remain outside the scope of this percolation study.

Under fiber-incomplete restriction, identical cluster cardinalities can arise from different transport spans (Section 3.5), so cluster size alone does not determine the generating set. The fiber structure of Section 2.6 makes the point at the generator level: four bytes share one transport value and are indistinguishable from the chirality register alone, so transport measurement alone does not identify which generator acted. This many-to-one map from generators to observables corresponds to the structure interpretability research calls superposition, where one measured quantity is consistent with more than one underlying feature. The same code family also carries a verified Hilbert-space lift with graph-state factorization into Bell pairs and CHSH values at the Tsirelson bound (`docs/reports/hQVM_Features_Report.md`, Formal Quantum Certification). Whether methods applied to models trained on Appendix A tasks recover transport rank or only squared cluster observables remains an open evaluation question.

Transport rank is collective: no single byte carries it, and the parity obstruction in Section 3.4 gives large alphabets with matched local byte statistics but different global reachability. Byte operators and canonical words share the same 256 generators yet produce different reachability regimes at different composition depth (Section 4.1). Shell reflection s maps to 6 - s is global, while word-level confinement depends on the start anchor; Appendix A.4 states two hypotheses about how a trained model might represent that anchor dependence. These patterns are candidates for structural comparison with context-dependent behavior in learned systems; the appendix tasks specify how to test them.

## 2. Spinorial State Space and Generators

### 2.1 The State Space Omega and Its Product Structure

The hQVM kernel operates on a 24-bit carrier state (GENE_Mac), composed of two 12-bit components A12 (active gyrophase) and B12 (passive gyrophase). The carrier admits 2^24 bit patterns, but **Omega** is the set of 4096 states reachable within two byte steps from **rest** (`0xAAA555`), the kernel anchor on the maximal-chirality horizon. Omega is the complete holographic closure of the constitutional horizon |H|^2 = 64^2 and is the full state space of the model. Patterns outside Omega lie beyond the holographic closure.

Omega factorizes as a Cartesian product U x V of two 64-element factors, with |U| = |V| = 64. The mask alphabet C64 is the self-dual binary [12,6,2] code (`docs/Findings/Analysis_Gravity.md`, Appendix B.5). U and V are the 64-element cosets of C64 realized in the distinct active and passive 12-bit patterns on Omega. The map s maps to (u(s), v(s)) and is a bijection Omega to U x V. Fiber-complete byte restrictions act on the two factors symmetrically, which underlies the product-cluster theorem.

The two 64-element factors are the boundary sets of the product. The CGM names them the constitutional horizons, denoted H. The state space size is the square of the boundary size, giving the holographic identity:

```
|Omega| = |H|^2 = 64^2 = 4096
```

|H| is the cardinality of the constitutional horizon (64 states at each chirality extreme). |Omega| is the full reachable state count, the square of the horizon cardinality. It is what must be squared to produce the full space, and it is what loses dimension when the alphabet is restricted.

### 2.2 Shells and Horizons

**Shells** partition Omega by chirality weight. The six-bit chirality register and shell index are

```
chi(s) = popcount( pair-collapse(A12 XOR B12) )    on six dipole pairs
shell(s) = popcount( chi(s) )
```

State s carries active gyrophase A12 and passive gyrophase B12. chi(s) is the six-bit chirality register obtained by pair-collapsing the XOR of the two faces. shell(s) is its Hamming weight, ranging from 0 to 6.

The shell distribution follows the binomial law with populations C(6,k) * 64 for k = 0 through 6:

- Shell 0: 64 states (equality horizon, zero chirality)
- Shell 1: 384 states (bulk)
- Shell 2: 960 states (bulk)
- Shell 3: 1280 states (equatorial maximum)
- Shell 4: 960 states (bulk)
- Shell 5: 384 states (bulk)
- Shell 6: 64 states (complement horizon, maximal chirality)

Shells 0 and 6 are the **constitutional horizons**, each carrying 64 states. Shells 1 through 5 form the relational **bulk** (3840 states). The rest state `0xAAA555` is on shell 6 (complement horizon).

The two horizons are the factor H realized at the two chirality extremes, the root written at both ends of the chirality register. Shells 1 through 5 carry bulk transit; shells 0 and 6 are horizon anchors. Bulk penetration and shell-resolved connectivity C(k) are structural percolation observables (Sections 6.7 and 6.8).

Shell index k equals popcount(chi) throughout this document. `docs/Findings/Analysis_hQVM_Wavefunction.md` uses arch_shell = 6 - k, measuring depth from the complement horizon. Cross-document shell lists convert by k_wave = 6 - k_pop.

| k (this document) | Horizon or bulk | arch_shell (Wavefunction) |
|-------------------|-----------------|---------------------------|
| 6 | complement horizon (rest anchor) | 0 |
| 5 through 1 | bulk | 1 through 5 |
| 0 | equality horizon | 6 |

### 2.3 Bytes and the Spinorial Transition

A **byte** is an 8-bit label for one update operator, a permutation of Omega. The full pool has 256 bytes. Each byte b in {0,...,255} induces a permutation T_b on Omega. The update has two steps, first XORing the active gyrophase A12 with a 12-bit mask determined by b and then gyrating by complementing and swapping the two 12-bit components. This rule is the CGM **spinorial transition**.

Transcription relative to the micro-archetype reference is

```
intron(b) = b XOR 0xAA
```

The byte label b ranges over {0,...,255}. The constant 0xAA is the **micro-archetype** (GENE_Mic), the reference pattern for transcription. The intron is the transcribed payload structure of b relative to that reference.

The transition is a discrete gyrogroup composition in which the L-step applies the mutation by XORing the active gyrophase with the mask and the R-step performs the gyration by complementing and swapping the two 12-bit components. A single byte implements one [L][R] modal step, a spinorial half-cycle that maps shell s to shell 6 - s, inverting chirality without returning to the rest reference in one step. It is the unclosed square root of the dynamics. The depth-four canonical word composes two gyrations, squaring the half-cycle to close the invariant loop.

### 2.4 CGM Stage Labels in the Byte

The Common Governance Model names four stages in the constitutional update cycle: **CS** (Common Source), **UNA** (Unity Non-Absolute), **ONA** (Opposition Non-Absolute), and **BU** (Balance Universal). In the hQVM kernel these names label positions in the 8-bit intron after transcription and in the two-step byte transition. The mapping from stage to bit positions follows `docs/specs/hQVM_Specs_Formalism.md`, Sections 1 and 2.

| CGM stage | Gyrogroup bit pair | Intron positions | Role in one byte step |
|-----------|-------------------|------------------|------------------------|
| CS | L0 (Left Identity) | 0, 7 | Family boundary. Selects which of the four K4 gate variants implements the update. |
| UNA | LI (Left Inverse) | 1, 6 | Payload bits that XOR the active gyrophase A12 with the 12-bit mask. |
| ONA | FG (Forward Gyration) | 2, 5 | Payload bits for the forward frame of the complement-and-swap gyration. |
| BU | BG (Backward Gyration) | 3, 4 | Payload bits at the fold where forward and reverse intron readings meet. |

Each byte implements one [L][R] modal step. The L-step applies the mask (UNA content). The R-step gyrates (ONA and BU content, gated by the CS family bits). A string such as **ONA|BU** names the boundary between adjacent stages. The **connection 1-form chain** assigns a magnitude to each of the seven boundaries CS|UNA, UNA|ONA, ONA|BU, BU|BU, BU|ONA, ONA|UNA, and UNA|CS.

### 2.5 Byte Classification Axes

The CGM fiber-bundle structure of the byte provides multiple classification axes (`docs/specs/hQVM_Specs_Formalism.md`, Sections 2 through 4).

**Family** (2 boundary bits, positions 0 and 7 of the intron). Four families of 64 bytes each, corresponding to the four elements of the K4 gauge group, namely id for family 00, S for family 01, C for family 10, and F for family 11.

**K4 gate.** Classification by the Klein four-group element determined by the boundary bits. The four gates are id (identity), S (pole swap via family 01), C (pole swap via family 10), and F (Z2 carrier flip). Each gate class contains 64 bytes.

**Q6 transport class** (6 payload bits, positions 1 through 6 of the intron). The map from byte to q6(b) is 4-to-1, giving 64 classes of 4 bytes each. A class at value v comprises the id and F bytes at payload v together with the S and C bytes at payload v XOR epsilon_6, where epsilon_6 = 63 is the all-ones six-bit vector (Section 3.4, Table 1, weight-6 layer). The family boundary bits (Section 2.4) determine which of these two payloads a given byte's transport value points to. The Q6 weight is popcount(q6(b)), ranging from 0 to 6, with distribution {0:4, 1:24, 2:60, 3:80, 4:60, 5:24, 6:4}. The value q6(b) in GF(2)^6 is the chirality transport value. It is the per-byte increment applied to the chirality register under the transition.

**Fold disagreement.** For each of four CGM phase pairs, compare forward and reverse readings of the palindromic byte. Fold disagreement counts how many pairs disagree. Distribution: {0:16, 1:64, 2:96, 3:64, 4:16}.

**Phase-net vector.** A 4-tuple (CS_xor, UNA_xor, ONA_xor, BU_xor) recording which CGM phases carry nonzero net XOR across the fold. There are 16 distinct phase-net vectors, each associated with 16 bytes.

**Curvature 2-form** at the BU fold. |F|^2 at the fold boundary between intron bits 3 and 4, a discrete holonomy defect on the BU stage boundary. Distribution: {0.0:64, 0.0625:128, 0.25:64}.

**Connection 1-form chain.** Seven boundary magnitudes at the constitutional phase boundaries above. There are 81 distinct connection chain signatures.

### 2.6 Fiber-Complete Generator Restrictions

Each byte b has a chirality transport value q6(b) in GF(2)^6. Four bytes sharing one transport value q form the **fiber** over q. A micro-reference groups the four family variants at one payload m; because id and F carry q = m while S and C carry q = m XOR epsilon_6, a payload group spans two distinct transport values.

A generator restriction A is **fiber-complete** when, for every transport value q represented in A, all four bytes with that value are included. This is percolation on the quotient by the four-fold family fiber over each transport class. Including the full fiber ensures that payload increments act symmetrically on the conjugate factors U and V. Fiber-complete restrictions produce clusters whose size equals the square of the reached factor cardinality. Fiber-incomplete restrictions break the product form.

This is completeness at the level of a single transport value: for the value to be included fully, all four bytes carrying it must be present. A coarser completeness holds at the level of one payload: the four family variants of a single micro-reference (Section 2.7) are present together as a group. A payload group's four bytes carry two distinct transport values (Section 2.5). Payload-complete restriction yields the product-cluster structure of Theorem 3.1 through a rank relation adapted to this coarser completeness.

A **micro-reference** is the six-bit payload index in the byte (intron bits 1 through 6) that selects one of 64 mask patterns. Word operators use label m for this index.

### 2.7 Canonical Word Operators

Canonical word operators compose bytes into closed invariants. From `docs/Findings/Analysis_hQVM_Wavefunction.md`, Theorems T1 through T10:

**Modal depth convention.** One byte implements one [L][R] pair (modal depth 2). W2 and W2' are two-byte half-words (modal depth 4). F = W2 then W2' is a four-byte full word (modal depth 8).

**W2(m).** Two-byte half-word [byte(fam 00, m), byte(fam 01, m)] at modal depth 4. Involution mapping shell s to 6 - s (Theorem T2).

**W2'(m).** Two-byte half-word [byte(fam 10, m), byte(fam 11, m)] at modal depth 4. Involution mapping shell s to 6 - s (Theorem T3).

**F(m).** Four-byte full word F = W2 then W2'. Preserves shell while flipping the carrier Z2 coordinate (Theorem T4).

All three are involutions, satisfying W2^2 = W2'^2 = F^2 = id on all 4096 states (Theorem T6). The set {id, W2, W2', F} forms a Klein four-group for every micro-reference m (Theorem T1). This word-level K4 is involutive closure bookkeeping at depth four, not the spinorial operator group.

Single-byte permutations T_b have order 4 on Omega for 126 of 128 distinct byte actions. The lifted operator family has odd-parity elements of order 4 whose squares lie in the diagonal center. Percolation on Omega sees the 128-class quotient; the order-4 byte structure is the discrete source of depth-four closure.

A single byte acts as a gyration, a half-cycle that inverts chirality without returning to the rest reference in one step. W2 composes two bytes and squares the half-cycle to a depth-four closure. F composes W2 with W2' and returns the carrier to the same pole with a Z2 sheet flip. Word operators are the closed invariants that preserve ancestry, while byte operators are the unclosed roots.

**L-type operators** (percolation label) are single-byte transition permutations T_b. Each byte implements one [L][R] modal step, where the L-step mutates the active gyrophase and the R-step performs gyration. L-type operators provide bulk access across all seven shells when q-diversity is sufficient.

**R-type operators** (percolation label) are canonical half-words and full words W2, W2', and F built from alternating family compositions. They realize the [L][R][L][R] versus [R][L][R][L] closure required by BU-Egress. From horizon-anchored starts, R-type operators confine reachability to constitutional poles (shells 0 and 6).

### 2.8 Percolation Events

**Reach(A)** is the reachable set from rest under generator set A. |Reach(A)| is its cardinality. Monte Carlo tables report E[|Reach| | nz] as the mean cardinality over samples with nonempty reachability.

**E_span** (horizon spanning, also constitutional spanning). From rest on shell 6, Reach(A) includes at least one state on shell 0.

**E_full** (full reachability). The reachable set from rest equals all 4096 states.

**E_pair.** Complete pole pairing under word dynamics. Every complement-horizon state has an equality-horizon partner in the word-reachable set (Theorem T9 set-to-set formulation).

**E_hit_eq_word.** Word-path from rest reaches any equality-horizon state. This is the weak word event, distinct from E_pair.

**E_spectrum.** Transport-curvature spectrum completion. All seven defect weights 0 through 6 appear among pairwise Q6 transport defects of the allowed byte set.

**E_horizon_confined.** Multi-source breadth-first search from all 64 complement-horizon states, or all 64 equality-horizon states, visits only states on that horizon. This event tests stabilizer structure of the active generator set.

## 3. The Square-Root Cluster Theorem

### 3.1 Statement

The CGM spinorial transition rule induces an affine GF(2)^d action on the chirality register (Steps 1 and 2 of the proof). The transport increment q_d(b) is determined by the byte payload, and commutation of byte pairs collapses to equality of q_d values. This structure follows from the byte update rule. Fiber-completeness is percolation on the quotient by the four-fold family fiber over each transport value, so the same affine action applies on the conjugate factors U and V.

Let A be a fiber-complete generator restriction. Define the **transport rank**:

```
r(A) = rank_GF(2)( span{ q_d(b) : b in A } )
```

where q_d(b) is the d-bit chirality transport value of byte b, and the span is taken in the vector space GF(2)^d over all bytes in the allowed set A. Define the **root dimension**

```
root(A) = 2^r(A)   for r(A) >= 1
root(A) = 2        for r(A) = 0
```

root(A) is the cardinality of the surviving transport subspace on each marginal factor. At rank zero only the gauge doublet survives, so root(A) = 2.

**Theorem (Square-Root Cluster).** For every fiber-complete generator restriction A, the reachable set from the rest anchor has marginal factors U_R and V_R in the factorization Omega = U x V. The factor cardinalities satisfy |U_R| = |V_R| = root(A). For r(A) at least 1,

```
|Reach(A)| = root(A)^2 = (2^r(A))^2
```

|Reach(A)| is the cardinality of the BFS reachable set from rest. U_R and V_R are the reached subsets of the marginal factors U and V respectively. At r(A) = 0, root(A) = 2, |Reach(A)| = 2, and rectangularity is 0.5.

The root of the cluster is root(A). The cluster is the holographic square of this root for r(A) at least 1. When root(A) = |H| = 64, equivalently r(A) = 6, the full root is recovered and |Reach| = |Omega| = 4096. When root(A) is smaller, ancestry is partially lost and the cluster shrinks as the square of the surviving root dimension.

**Corollary (normalized coverage).** Under fiber-complete restriction, the coverage fraction

```
|Reach(A)| / |Omega_d| = (root(A) / |H_d|)^2 = (2^r(A) / 2^d)^2
```

for r(A) at least 1. Cluster geometry, transport rank, and expected coverage share one normalized object: the square of the root fraction on each conjugate face.

### 3.2 Proof

The proof connects three ingredients, namely the per-byte transport rule on the chirality register, the product structure Omega = U x V, and breadth-first search reachability from rest.

**Step 1. Chirality transport is an affine GF(2)^d action.** The chirality register is chi(s) = pair-collapse(A12 XOR B12). Under a byte transition T_b, the active gyrophase is mutated by the mask mu(b) and the two gyrophases are complemented and swapped. The chirality transforms as:

```
chi(T_b(s)) = chi(s) XOR q_d(b)
```

T_b is the global permutation induced by byte b on state s. The increment q_d(b) is the d-bit transport value determined by the payload bits of b. The chirality register therefore evolves by XOR addition in GF(2)^d, independent of the family (gauge) bits.

**Step 2. The transport root is the affine span of the allowed values.** From the rest anchor, the reachable chirality values form the affine coset:

```
chi(Reach(A)) = chi(rest) + Q(A),   Q(A) = span_GF(2){ q_d(b) : b in A }
```

Q(A) is the linear transport subspace spanned by allowed q_d values. chi(rest) is the chirality at the rest anchor. The reachable chirality set is the affine coset chi(rest) + Q(A). It contains 2^r(A) values. The set of reachable chirality values is a coset of Q(A) and has the same cardinality 2^r(A).

**Step 3. The product structure splits the root across the two factors.** The map s maps to (u(s), v(s)) is a bijection Omega to U x V. Each factor is a 2^d-element C64 coset. The chirality chi(s) is recovered from the pair (u(s), v(s)). A fiber-complete restriction includes, for every transport value q present in A, all four family bytes carrying q. The four family bytes act symmetrically on the two factors, because the family bits select the gauge variant of the same payload increment. Under fiber-completeness, the action of A on U and on V is the same affine action by Q(A).

**Step 4. The reachable factors are affine subspaces of dimension r(A).** Within each factor, the reachable set is the affine subspace reached by applying the increments Q(A) to the rest coordinate. Its cardinality equals |Q(A)| = 2^r(A). Symmetry across factors gives |U_R| = |V_R| = 2^r(A), and the product structure is preserved:

```
Reach(A) = U_R x V_R
```

U_R and V_R are the reached subsets of the marginal factors U and V under generator set A.

**Step 5. The cluster size is the square of the root.** Taking cardinalities:

```
|Reach(A)| = |U_R| * |V_R| = 2^r(A) * 2^r(A) = (2^r(A))^2
```

|U_R| and |V_R| are the cardinalities of the reached marginal factors. Their product is the cluster cardinality under fiber-completeness.

This completes the proof. The cluster is a perfect square because ancestry preservation forces a conjugate product structure and the root acts symmetrically on both factors under fiber-completeness.

This derivation depends only on the affine transport rule and the product structure Omega = U x V, both of which hold for every chirality dimension d in the hQVM(d) family. The proof is unchanged under d, with GF(2)^6 replaced by GF(2)^d throughout.

### 3.3 The Root Dimension and the Holographic Square

The theorem restates the holographic identity |Omega| = |H|^2 at every level of restriction. The full state space corresponds to r = 6, root = 64, cluster = 4096. Partial restriction lowers r and shrinks the cluster as the square of the surviving root:

| r(A) | root(A) | |Reach(A)| | Interpretation |
|------|---------|------------|----------------|
| 0 | 2 | 2 | Zero-transport gauge doublet (Table 1) |
| 1 | 2 | 4 | One transport direction |
| 3 | 8 | 64 | Q6 weight-6 layer alone |
| 4 | 16 | 256 | Single phase-net vector |
| 5 | 32 | 1024 | Even-weight subspace |
| 6 | 64 | 4096 | Full root, full state space |

Every row in the structured case table matches the predicted cluster size from transport rank. The square-root column is the measure of preserved ancestry. The cluster shrinks as the square of the lost root dimension when r(A) decreases from 6.

Across the hQVM(d) family for d = 1 through 8, 52 of 52 fiber-complete weight and cumulative layers match breadth-first reachability exactly.

This is the functional universality identity in percolation form. Connectivity is controlled by the transport rank r(A) of the allowed generator set. Spanning, full coverage, and intermediate plateaus form one coverage hierarchy on the same restricted generator graph; rank sets cluster size at each plateau, and p_c marks the onset of each event.

### 3.4 Parity Obstruction

The even-weight transport subspace illustrates a rank-5 plateau. The set of bytes whose q6 value has even popcount spans a 5-dimensional subspace of GF(2)^6. Its 128 bytes have maximum rank 5 and maximum cluster 1024 = 32^2. Adding sufficient odd-weight bytes restores rank 6 and full connectivity.

**Corollary (parity obstruction).** A generator restriction that contains only even-weight transport values reaches only even shells. The even-weight subspace has rank 5 and reaches 1024 states. Generator restrictions with odd-weight transport capability can reach all shells, and the odd-weight byte set reaches all 4096 states.

Chirality popcount changes parity under an odd-weight increment and is preserved under an even-weight increment. From rest on shell 6, which has even popcount 6, only even shells remain reachable when every increment is even. Full coverage of all seven shells therefore requires at least one odd-weight transport value, which lifts the rank to 6 once the span is otherwise full.

**Case r = 0.** When the allowed alphabet contains only bytes with q6 = 0, the transport span is {0} and r(A) = 0. The four weight-0 bytes (one per family) share a zero payload and differ only in gauge phase. Their gauge action on rest produces the 2-state swap doublet {(u, v), (v, u)}, both on shell 6, with |U_R| = |V_R| = 2 and rectangularity 0.5. The measured cluster is 2 (Table 1). The deterministic gate confirms this prediction.

**Generalization across d.** The even-weight transport subspace has dimension d - 1 at every d, since parity is a fixed linear functional on GF(2)^d and its kernel is always a hyperplane. At even d, the even-weight restriction reaches exactly (2^(d-1))^2 states, matching Theorem 3.1 without correction, as at d = 6. At odd d, the same restriction reaches exactly twice this count. The distinction tracks the parity of d rather than any property specific to d = 6: epsilon_d, the all-ones transport vector, has weight d, so it lies inside the even-weight subspace when d is even and outside it when d is odd. Whether this placement is the direct cause of the factor of two, through a failure of the symmetric U/V action required in Step 3 of the proof of Theorem 3.1, is a natural next question; it has not been derived here and is stated as an open structural observation, verified for d = 1 through 8.

### 3.5 Fiber-Incompleteness

The fiber-complete condition defines when the product-cluster structure holds. Selecting individual bytes without their family partners breaks the symmetric correspondence between the U and V factors. Six bytes with independent transport values achieving rank 6 reach 4064 states. Fiber completeness ensures each transport value contributes symmetrically to both factors, so the product form is preserved and the rank determines the cluster size.

The theorem therefore gives a sharp dichotomy. Under fiber-complete restriction, cluster size matches the transport-rank prediction. Under fiber-incomplete restriction, the reachable set may be a proper subset of the product, and rank-6 selections reach up to 4064 states with rectangularity below 1, so the product form depends on the fiber quotient rather than on symmetric definition alone. This is the structural reason why random byte subsets reach full Omega in expectation at sufficient p, while structured fiber-complete alphabets hit the square plateaus precisely.

### 3.6 The Square Root in Percolation Formalism

In standard bond percolation, cluster nodes are fundamental scalars and the order parameter P_infty(p) measures the probability that a node belongs to the giant cluster. Square roots enter only as emergent statistics, appearing as the root-mean-square displacement of random walkers and as exact critical exponents such as sqrt(17) in corner percolation above.

This product state space changes the location of the square root because the node set Omega is a holographic product U x V, so the cluster size is a square by construction. The standard percolation order parameter, cluster size, is therefore the square of an underlying transport dimension. An external observer who measures a cluster of 4096 states on Omega is measuring a 64-dimensional transport root, squared by the requirement of bipartite causal closure.

The bipartite structure is the discrete expression of ancestry preservation, in which a forward action on the active gyrophase must be bound to its conjugate on the passive gyrophase to form an invariant. The product U x V is the algebraic form of this binding, and squaring is the cost of producing an observable from a directed half-path. This is the same algebraic maneuver that produces observables in relativistic quantum mechanics through the squared energy-momentum invariant and in quantum measurement through the squared amplitude, recorded here as a structural parallel.

### 3.7 Computational Validation

Deterministic verification gates cover the structured case table at d = 6 (every Q6 weight layer, cumulative layer, even-weight subspace, and family-complete alphabet) and the hQVM(d) family for d = 1 through 8. Each case reports rank r(A), BFS reachable set size, and PASS or FAIL against the predicted cluster size. The structured case table is the executable form of the theorem.

## 4. The Two Operator Regimes: Roots and Squares

The byte and word regimes apply the same 256 byte operators at different levels of the multiplicative structure. **L-type** single-byte steps act on the full product U x V and connect maximally when transport diversity is sufficient. **R-type** canonical four-byte words act within the root H and confine reachability from horizon anchors. This section reports percolation under each regime.

### 4.1 Word Confinement as Root-Level Action

**Theorem (word confinement).** Under any composition of canonical word operators W2, W2', F at any set of micro-references, the reachable set from rest is confined to at most 128 states on the two constitutional horizons.

**Argument.** W2(m) maps shell s to 6 - s (Theorem T2). F(m) preserves shell (Theorem T4). From shell 6, W2 moves to shell 0 and further W2-type steps alternate between shells 0 and 6. F acts within a shell. No word operator changes shell by 1 through 5. Each W2(m) is an involution pairing horizon states across shells 0 and 6. Word reachability from rest is connectivity in the union of selected pairings on those 128 nodes. The 128 states are the two copies of the root H, written at shells 0 and 6.

The confinement is the statement that closure operators act within the root at shells 0 and 6. Bulk access from rest requires byte operators or a changed anchor. Shuffling the label-to-dynamics map severs the correspondence between operator labels and root structure, restoring bulk access. The confinement is anchor-dependent, as the breadth-first search over all 64 W2(m) operators shows.

Word-regime breadth-first search using all 64 W2(m) operators, by start anchor:

| Start anchor | W2-BFS reach | Shells hit |
|--------------|-------------|------------|
| Rest (shell 6) | 128 | {0, 6} |
| Bulk (shell 2) | 128 | {2, 4} |
| Equality horizon (shell 0) | 128 | {0, 6} |

From horizon anchors, word dynamics visits only the two constitutional horizons. From bulk anchors, W2 reaches the antipodal bulk shell pair, because the global rule shell s maps to 6 - s carries the copy of the root at the anchor into the copy at the reflected shell. The observable confinement therefore depends on which copy of the root the anchor occupies, while the root-level pairing itself is global.

### 4.2 Exact Deterministic Reachability (Byte Regime)

Exact deterministic reachability from the rest state for structured generator restrictions follows from breadth-first search to depth 12. Depth 12 exceeds the longest canonical word path (modal depth 8 for F) and accommodates restricted semigroups that need more than two steps without saturating early. Supplementary deterministic checks are indexed in Appendix B.

#### 4.2.1 Family Restrictions

Each of the four families alone (64 bytes) reaches all 4096 states and achieves constitutional spanning at depth 1. All pairs and triples of families do the same. Percolation thresholds vanish at the family level, because any single family already reaches full Omega, and the system is therefore maximally connected under any single family.

Each family of 64 bytes generates a transitive action on Omega. The K4 gauge structure partitions the 256 bytes into four equally powerful subgroups, each sufficient for global reachability. Each family is fiber-complete over its 64 transport values, with rank 6, so the square-root theorem gives cluster 4096 directly.

#### 4.2.2 K4 Gate Restrictions

Each of the four K4 gates alone (id, S, C, F, with 64 bytes each) reaches all 4096 states at depth 1. The shell-preserving gates (id + F, 128 bytes) and the pole-swap gates (S + C, 128 bytes) both reach all states.

Pole-swap capability is unnecessary for spanning at the byte level. The id gate alone, which applies no complement during gyration, still generates a transitive action on Omega because the mutation step (XOR with the 12-bit mask) provides sufficient chirality variation.

#### 4.2.3 Q6 Weight Restrictions

| Generator set | Bytes | Reach | Shells | Span | Giant | Span@D |
|----------|-------|-------|--------|------|-------|--------|
| Q6 weight=0 | 4 | 2 | (6) | No | No | - |
| Q6 weight=1 | 24 | 4096 | (0-6) | Yes | Yes | 6 |
| Q6 weight=2 | 60 | 1024 | (0,2,4,6) | Yes | No | 3 |
| Q6 weight=3 | 80 | 4096 | (0-6) | Yes | Yes | 2 |
| Q6 weight=4 | 60 | 1024 | (0,2,4,6) | Yes | No | 3 |
| Q6 weight=5 | 24 | 4096 | (0-6) | Yes | Yes | 6 |
| Q6 weight=6 | 4 | 4 | (0,6) | Yes | No | 1 |

Table 1. Reachability by Q6 weight layer. Span@D is the first depth at which constitutional spanning occurs.

Even-weight layers (0, 2, 4, 6) reach only even shells. Odd-weight layers (1, 3, 5) reach all seven shells and generate giant components. This parity obstruction caps fiber-complete clusters at 1024 = 32^2 when only even q6 values are present.

The four bytes with Q6 weight 6 (full chirality inversion) each connect rest to the equality horizon at depth 1, reaching four states including both horizons. The weight-6 layer alone has rank 1, so the square-root theorem gives cluster (2^1)^2 = 4, matching the census.

Cumulative Q6 layers (weight at most k) reach full Omega at k at least 1. The 28 bytes with weight 0 or 1 suffice.

#### 4.2.4 Fold Disagreement Restrictions

| Alphabet | Bytes | Reach | Shells | Span | Giant | Full |
|----------|-------|-------|--------|------|-------|------|
| Fold disagree=0 | 16 | 64 | (0,2,4,6) | Yes | No | No |
| Fold disagree=1 | 64 | 4096 | (0-6) | Yes | Yes | Yes |
| Fold disagree=2 | 96 | 4096 | (0-6) | Yes | Yes | Yes |
| Fold disagree=3 | 64 | 4096 | (0-6) | Yes | Yes | Yes |
| Fold disagree=4 | 16 | 256 | (0,2,3,4,6) | Yes | No | No |

Table 2. Reachability by fold disagreement level. Cumulative unions appear in Appendix C.

The 16 flat bytes (zero fold disagreement) reach 64 states across even shells and achieve constitutional spanning. Levels 1 through 3 alone each generate full Omega. Level 4 alone reaches 256 states. Adding any byte with nonzero fold disagreement (80 bytes total for disagree at least 1) restores full Omega reachability.

Fold disagreement classifies palindromic phase readings and is independent of Q6 transport weight. The level-4 alphabet therefore reaches shell 3 because its bytes carry odd-weight transport values, consistent with the parity obstruction above.

#### 4.2.5 Curvature 2-Form Restrictions

| |F|^2 at fold | Bytes | Reach | Span | Giant |
|--------------|-------|-------|------|-------|
| 0.000000 | 64 | 256 | Yes | No |
| 0.062500 | 128 | 4096 | Yes | Yes |
| 0.250000 | 64 | 1024 | Yes | No |

Table 3. Reachability by curvature 2-form magnitude.

The 64 bytes with zero curvature at the BU fold reach 256 states. The 128 bytes with |F|^2 = 0.0625 reach all 4096 states. The 64 bytes with |F|^2 = 0.25 reach 1024 states. Cumulatively, all bytes with |F|^2 at most 0.0625 (192 bytes) reach full Omega.

#### 4.2.6 Phase-Net Restrictions

Each of the 16 phase-net vectors is associated with 16 bytes. None generates a giant component alone. All achieve constitutional spanning. The full 16-row table is in Appendix C.1. Vectors with only CS active reach 64 states. All other single vectors reach 256 states. Requiring at least three active phases (80 bytes) restores full Omega.

#### 4.2.7 Connection 1-Form Boundary Analysis

| Boundary | Bytes with \|A\|>0 | Reach | Span | Bytes with \|A\|=0 | Reach | Span |
|----------|-------------------|-------|------|-------------------|-------|------|
| CS\|UNA | 192 | 4096 | Yes | 64 | 4096 | Yes |
| UNA\|ONA | 192 | 4096 | Yes | 64 | 4096 | Yes |
| ONA\|BU | 192 | 4096 | Yes | 64 | 256 | Yes |
| BU\|BU | 192 | 4096 | Yes | 64 | 256 | Yes |
| BU\|ONA | 192 | 4096 | Yes | 64 | 256 | Yes |
| ONA\|UNA | 192 | 4096 | Yes | 64 | 256 | Yes |
| UNA\|CS | 192 | 4096 | Yes | 64 | 4096 | Yes |

Table 4. Boundary necessity: excluding bytes with nonzero connection 1-form at each phase boundary.

The CS-pipe boundaries (CS|UNA and UNA|CS) are redundant for reachability. The 64 bytes with zero connection form at these boundaries still reach all 4096 states. The BU-fold boundaries (ONA|BU, BU|BU, BU|ONA, ONA|UNA) are necessary. Excluding bytes active at any of these boundaries collapses reachability to 256 states.

Exhaustive search over all 128 subsets of boundaries, where mask bit i indicates that boundary i may carry nonzero connection, finds that full Omega requires at least five active boundaries. The unique minimal mask at popcount 5 is {CS|UNA, ONA|BU, BU|BU, BU|ONA, ONA|UNA} with 64 bytes. A byte is allowed under mask m when its connection chain is zero on every inactive boundary and may be nonzero on active boundaries. Only five of 128 masks achieve full Omega. Popcount 6 and 7 also suffice with 64 and 256 bytes respectively.

This maps to the CGM stage hierarchy. The depth portion of the structure (ONA through BU and back) generates full connectivity. The gauge-to-mutation interfaces (CS|UNA and UNA|CS) are redundant for single-boundary exclusion. CS|UNA appears in the minimal active set when all seven boundaries are constrained simultaneously.

Bytes with zero curvature at the BU fold reach 256 states (Table 3). Nonzero fold curvature is a prerequisite for global transitivity under boundary-masked restriction. The BU-fold channels supply the non-abelian content that prevents collapse into a flat 256-state subspace.

### 4.3 Probabilistic Percolation Sweeps

Monte Carlo sweeps draw independent inclusion of groups (families, bytes, Q6 classes, or micro-references) with probability p per group. Each sample uses breadth-first search from rest to depth 12. Sample count n = 300 per sweep.

#### 4.3.1 Byte-Fraction Sweep

| p | P(full) | E[R \| nz] |
|---|---------|------------|
| 0.01 | 0.000 | 275 |
| 0.02 | 0.173 | 1267 |
| 0.03 | 0.547 | 2797 |
| 0.04 | 0.803 | 3519 |
| 0.05 | 0.913 | 3844 |
| 0.06 | 0.973 | 4017 |
| 0.07 | 1.000 | 4096 |
| 0.10 | 1.000 | 4096 |

Table 5. Byte-fraction percolation (256 independent byte groups, n = 300). P(full) is unconditional over all samples. E[R | nz] is mean reachable set size conditional on a nonempty generator set. Spanning thresholds from the same sweep are p_c(span) approximately 0.0215 and p_c(full) approximately 0.0288 (`hqvm_percolation_analysis_results.txt`, Section VIII).

The weak and strong events have distinct thresholds, with p_c(span) approximately 0.0215 and p_c(full) approximately 0.0288 (about 7.4 expected bytes at the strong threshold). Constitutional spanning occurs at lower p because hitting shell 0 requires fewer bytes than covering all seven shells. At p = 0.02, P(full) is approximately 0.17. The strong transition sharpens between p = 0.03 and p = 0.06.

#### 4.3.2 Transitive Expansion and Threshold Separation

Percolation on finite transitive graphs is governed by the isoperimetric profile of the active subgraph, together with bond occupancy (Alon, Benjamini, and Stacey, 2004). The Cheeger constant measures the ratio of boundary size to volume for vertex subsets. A high Cheeger constant forces rapid merger of small clusters into a giant component.

The hQVM byte-action graph exhibits strong expansion at the structural level. Any single family of 64 bytes generates a transitive action on all 4096 states. The strong full-Omega threshold p_c approximately 0.029 requires about 7.4 expected bytes. The separation between the weak spanning threshold 0.022 and the strong threshold 0.029 quantifies threshold separation on the byte graph. The gap of 0.007 reflects the seven-shell coupon-collector structure on the six-bit chirality register.

The canonical-word regime operates on the subgraph induced by depth-4 closure operators. From rest, reachability confines to 128 horizon states. This subgraph has poor expansion relative to the byte graph. Shuffling the byte-to-dynamics mapping while preserving micro-reference labels destroys word-regime confinement and restores full-Omega reachability. Confinement depends on the classification-to-dynamics correspondence. Transitive byte action and the 0.007 gap between p_c(span) and p_c(full) are the measured expansion signatures on the byte-action graph.

#### 4.3.3 Syntactic Spanning versus Semantic Reachability

Constitutional spanning and full-Omega reachability are distinct algebraic events. While spanning requires only a path from shell 6 to shell 0, which can occur with aligned mutations along a low-dimensional q6 subspace, full reachability requires r(A) = 6 together with odd-shell transport capability. The gap between p_c(span) approximately 0.022 and p_c(full) approximately 0.029 is the additional cost of orthogonal q6 diversity beyond pole acknowledgment.

#### 4.3.4 Family-Fraction Sweep

Here p is the probability of including each of the four families. Each family alone reaches full Omega. The full-Omega threshold is p_c approximately 0.15, matching the coupon-collector baseline p_null = 1 - 0.5^(1/4) approximately 0.159 for four independent groups.

#### 4.3.5 The Two Register Protocols: Q6-Class and Micro-Reference

**Q6-class restriction** includes each of the 2^d transport values independently with probability p, bringing all four bytes carrying that value at once. This is value-complete by construction (Section 2.6), so Theorem 3.1 applies directly: the rank of the included values is the rank of a random subset of GF(2)^d under independent Bernoulli(p) inclusion of its 2^d points. At d = 6, the exact rank threshold is p_c approximately 0.1053, with excess coordinate z = 2^d p - d equal to approximately 0.740 at onset.

**Micro-reference restriction** includes each of the 2^d payloads independently with probability p, bringing the four family variants of that payload at once. Because the id and F families leave the transport value at the included payload m unchanged, while the S and C families shift it to m XOR epsilon_d (Section 2.5), a single included payload contributes two transport values, not one:

```
q_d(G(m)) = {m, m XOR epsilon_d},   epsilon_d = 2^d - 1
```

where G(m) denotes the four-byte micro-reference group at payload m. Every included payload therefore brings epsilon_d into the transport span for free.

**The quotient reduction.** The map pi that sends m and m XOR epsilon_d to the same class is well defined because epsilon_d generates a subgroup of order two in GF(2)^d. The quotient GF(2)^d / {0, epsilon_d} is itself a GF(2) vector space, of dimension d - 1, since a quotient by a one-dimensional subspace loses exactly one dimension. It has 2^(d-1) classes, each the image of exactly one payload pair {m, m XOR epsilon_d}.

Let M denote a set of included payloads, and let r(M) be the rank of the transport values generated by M, as in Theorem 3.1. Because every included payload brings epsilon_d into the span, r(M) decomposes as

```
r(M) = [1 if M is nonempty, else 0] + rank_{GF(2)^(d-1)}( pi(M) )
```

where pi(M) is the image of M under the quotient projection, and the second rank is computed inside the (d-1)-dimensional quotient space.

**Bridge to reachability.** Combining this decomposition with Theorem 3.1 gives, for the micro-reference protocol:

**Corollary (rank-reachability equivalence).** Under micro-reference restriction, the reachable set from rest equals all of Omega_d if and only if r(M) = d.

**The percolation process.** Under independent payload inclusion with probability p, a quotient class {m, m XOR epsilon_d} is activated, meaning its image enters pi(M), exactly when at least one of its two constituent payloads is included. Since the two payloads of a class are disjoint and independently sampled, the activation probability per class is

```
p_pair = 1 - (1 - p)^2
```

and activations are independent across the 2^(d-1) classes, because different classes draw from disjoint payload pairs. The rank rank_{GF(2)^(d-1)}(pi(M)) is therefore the rank of a random subset of a (d-1)-dimensional GF(2) vector space, each of its points included independently with probability p_pair.

**The scaling coordinate.** The expected number of activated classes at inclusion probability p_pair is 2^(d-1) p_pair. The natural excess coordinate, the expected number of activated classes above the (d-1) needed to span the quotient space, is

```
z_root = 2^(d-1) p_pair - (d - 1) = 2^(d-1) ( 1 - (1-p)^2 ) - (d - 1)
```

This plays the role for the micro-reference protocol that the plain excess z = 2^d p - d plays for the Q6-class protocol. The factor-of-two reduction in effective dimension is the algebraic signature of the payload pairing.

**Exact rank distribution.** The probability that a random subset of an n-dimensional GF(2) vector space, each of its 2^n points included independently with probability q, has rank exactly k, is given in closed form by Mobius inversion on the lattice of GF(2) subspaces (Fulman and Goldstein, 2014; MacWilliams and Sloane, 1977):

```
P(rank = k) = [n choose k]_2 * sum_{j=0}^{k} mu(j,k) [k choose j]_2 (1-q)^(2^n - 2^j)
```

where [n choose k]_2 is the Gaussian binomial coefficient and mu(j,k) = (-1)^(k-j) 2^((k-j)(k-j-1)/2) is the Mobius function of that subspace lattice.

Applying this with n = d - 1 and q = p_pair gives the exact distribution of the quotient rank, and through the decomposition above, the exact distribution of the full transport rank r(M), and through Theorem 3.1, the exact distribution of the reachable set size.

**Exact equation of state.** Write theta(p, d) for the expected coverage fraction E[|Reach(M)|] / |Omega_d|. Since |Reach(M)| = (2^r(M))^2 for r(M) at least 1 and |Reach(M)| = 2 at r(M) = 0, theta is a finite sum over the exact rank distribution:

```
theta(p, d) = sum_{k=1}^{d} P(r(M) = k) (2^k)^2 / 2^(2d) + P(r(M) = 0) * 2 / 2^(2d)
```

Its derivative with respect to z_root,

```
chi(z_root, d) = d theta / d z_root
```

is the susceptibility of the coverage transition. Under micro-reference restriction, P(rank = k), theta(p, d), chi, and p_c(d) are exact functionals of the transport rank distribution.

**Asymptotic threshold.** Let p_c(d) be the value of p at which P(r(M) = d) = 1/2, and let z_root,c(d) be z_root evaluated at p_c(d). This sequence converges as d grows:

```
z_root,c(d) = c_inf - a/d + o(1/d),   c_inf approximately 1.2665,   a approximately 1.1385
```

Equivalently, to leading order for large d,

```
p_c(d) approximately 1 - sqrt( 1 - (d - 1 + c_inf) / 2^(d-1) )
```

At d = 6 the exact threshold is p_c approximately 0.0908; the asymptotic formula evaluates to approximately 0.10 at that d, reflecting the 1/d finite-size correction visible through d = 16 in the scaling analysis (Appendix B).

| d | p_c(rank), exact | z_root,c(d) |
|---|------------------|-------------|
| 4 | 0.2188 | 0.118 |
| 5 | 0.1458 | 0.324 |
| 6 | 0.0908 | 0.547 |
| 7 | 0.0541 | 0.740 |
| 8 | 0.0313 | 0.888 |

The Q6-class threshold uses the same rank distribution with n = d and q = p, giving at d = 6 an exact p_c approximately 0.1053 with z = 2^d p - d equal to approximately 0.740 at onset. Under both protocols, P(full Omega) and P(rank = d) coincide for d at most 4 by exhaustive payload enumeration and for d at most 8 through the exact rank distribution.

### 4.4 Shell-Resolved Thresholds

| p | Shell 0 | Shell 1 | Shell 2 | Shell 3 | Shell 4 | Shell 5 | Shell 6 |
|---|---------|---------|---------|---------|---------|---------|---------|
| 0.005 | 0.045 | 0.155 | 0.350 | 0.385 | 0.330 | 0.155 | 0.735 |
| 0.010 | 0.135 | 0.355 | 0.605 | 0.700 | 0.620 | 0.445 | 0.930 |
| 0.020 | 0.455 | 0.755 | 0.910 | 0.915 | 0.915 | 0.785 | 0.995 |
| 0.030 | 0.775 | 0.950 | 0.995 | 0.995 | 0.995 | 0.955 | 1.000 |
| 0.050 | 0.970 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 |

Table 6. Shell-resolved percolation (n = 200): probability that each shell appears in the reachable set from rest. Shell 6 entries fall below 1.0 at low p because empty generator sets contribute zero probability mass to all shells.

Shell 0 (equality horizon) is the bottleneck for constitutional spanning. Bulk shells become reachable at lower p than shell 0, in line with the binomial population weights.

### 4.5 Minimum Spanning Alphabets

Four individual bytes span Omega from rest at depth 1:

| Byte | Family | Gate | Q6 weight | Fold disagree | Phase-net |
|------|--------|------|-----------|---------------|-----------|
| 0x2A | 10 | S | 6 | 1 | (1,0,0,0) |
| 0x55 | 11 | F | 6 | 0 | (0,0,0,0) |
| 0xAB | 01 | C | 6 | 1 | (1,0,0,0) |
| 0xD4 | 00 | id | 6 | 0 | (0,0,0,0) |

Table 7. Single-byte spanning set. All have Q6 weight 6.

A single byte achieves constitutional spanning if and only if it inverts all six chirality bits (Q6 weight 6). The four spanning bytes include one from each family. A weight-6 byte has transport value q6 = 111111, which gives rank 1, root 2, and the four-state cluster that touches both horizons.

Among 5000 random byte pairs, 205 pairs achieve constitutional spanning. Same-family, cross-family, same-gate, and cross-gate pairs all appear.

### 4.6 Payload-Axis Sweeps

The hQVM 24-bit carrier unpacks to a 2 x 3 x 2 tensor with six oriented dipole pairs (`docs/specs/hQVM_Specs_Formalism.md`, Section 2.1). Three pairs in Frame 0 correspond to rotational generators of se(3). Three pairs in Frame 1 correspond to translational generators. CGM derives six operational degrees of freedom at the ONA stage as the se(3) Lie algebra SE(3) = SU(2) semidirect R^3. Each payload bit flips one dipole pair, executing a discrete pi-rotation around the corresponding generator.

The three payload-axis labels are **LI** (Left Inverse, intron bit pairs 1 and 6), **FG** (Forward Gyration, pairs 2 and 5), and **BG** (Backward Gyration, pairs 3 and 4). Each axis sweep includes the 48 micro-reference groups whose payload touches that axis (192 bytes total per axis).

| Axis | p_c (full) | E[bytes] at p_c |
|------|------------|-----------------|
| LI | 0.117 | ~22.5 |
| FG | 0.115 | ~22.1 |
| BG | 0.118 | ~22.7 |

Table 8. Payload-axis percolation thresholds.

The three axes yield equivalent thresholds within Monte Carlo error. No single generator direction is privileged for byte-level connectivity. The three rotational and three translational degrees of freedom enter the transport root symmetrically.

### 4.7 Shadow-Pair Percolation

Each Q6 transport class contains 4 bytes (one per family). Bytes b and b XOR 0xFE are a **shadow pair** that induces the same permutation on Omega (`docs/Findings/Analysis_hQVM_Wavefunction.md`, Section 9). The XOR mask 0xFE flips the two family boundary bits while preserving the six-bit transport payload. Three inclusion modes were tested:

| Mode | Description | p_c (full) |
|------|-------------|------------|
| full_fiber | All 4 family bytes per selected Q6 class | 0.108 |
| one_repr | One representative byte per selected Q6 class | 0.111 |
| shadow_one | One byte per shadow pair per Q6 class | 0.105 |

Table 9. Q-fiber percolation thresholds by selection mode.

The three modes yield thresholds within approximately 0.006 of each other. Spinorial shadow redundancy leaves byte-level percolation at the Q6-class resolution unchanged within Monte Carlo error.

### 4.8 Horizon-Stabilizer Percolation

The four holonomic gate bytes HORIZON_GATE_BYTES = {0xAA, 0x54, 0xD5, 0x2B} are the unique bytes that preserve every complement-horizon state and every equality-horizon state in one step (Feature 121 in `docs/reports/hQVM_Features_Report.md`). This census identifies the set of one-step horizon stabilizers on all 64 states of each horizon.

Breadth-first search from all 64 complement-horizon states using only these four bytes reaches 64 states with E_horizon_confined = True.

Monte Carlo over random nonempty byte subsets gives a different picture:

| p | P(conf \| comp) | P(conf \| eq) | P(full) |
|---|-----------------|---------------|---------|
| 0.02 | 0.000 | 0.000 | 0.154 |
| 0.05 | 0.000 | 0.000 | 0.933 |
| 0.10 | 0.000 | 0.000 | 1.000 |

Table 10. Horizon confinement versus bulk percolation (conditional on nonempty generator set). P(conf) is the probability that all visited states remain on the starting horizon.

Any nonempty random generator set escapes horizon confinement immediately (P(conf) = 0 for all tested p greater than 0), while bulk percolation P(full) rises to unity by p approximately 0.10. The four stabilizer bytes confine dynamics when selected alone. Under uniform random generator set sampling they appear with probability (4/256)^k and horizon confinement is absent for k at least 1. Byte-level freedom and horizon stability are incompatible under random access.

### 4.9 Byte-Regime Null Models

Three byte-level null models were compared to the real CGM dynamics.

**Null 1 (shuffled family labels).** Permutations of the 256 bytes reassigned to four pseudo-families of 64. P(span) at p = 0.20 is 0.58 (real) versus 0.61 (shuffled average). Family label structure separates real from shuffled spanning by 0.03 at this p.

**Null 2 (random k-byte subsets).** Unstructured random subsets of k bytes versus k bytes drawn with family structure. For k at least 16, both reach P(span) = 1.0.

**Null 3 (shuffled byte-to-dynamics mapping).** Each byte retains its CGM labels but receives a random other byte's transition table. On the family-fraction sweep, real and shuffled P(full) agree within approximately 0.03 at p = 0.20 through 1.0. Byte-regime full-Omega percolation is robust to dynamics shuffling at the family level.

**Null 4 (word-regime dynamics shuffle).** Each byte keeps its micro-reference label but receives a random other byte's transition table. Word percolation (W2 + W2') is compared under real versus shuffled dynamics:

| p | Real P(full) | Shuffled P(full) | Gap |
|---|-------------|-----------------|-----|
| 0.05 | 0.000 | 0.060 | -0.060 |
| 0.10 | 0.000 | 0.458 | -0.458 |
| 0.15 | 0.000 | 0.883 | -0.883 |
| 0.20 | 0.000 | 0.992 | -0.992 |
| 0.30 | 0.000 | 1.000 | -1.000 |

Table 11. Word percolation under real versus shuffled dynamics.

Real CGM dynamics confines word reachability to 128 horizon states. Shuffled dynamics reaches full Omega by p = 0.30. Confinement requires the true classification-to-dynamics map.

Dixon (1969) motivates unstructured random permutations on n labels as a group-generation baseline. The CGM bytes are fixed global permutations with algebraic label structure. Nulls 1 through 4 test label and dynamics decoupling while preserving the true permutation group.

### 4.10 Constitutional Byte Shell Trace and Z2 Helix

The canonical 4-family turn at micro-reference 1 (bytes 0xA8, 0xA9, 0x28, 0x29) from rest produces:

| Step | Byte | Shell | Sector |
|------|------|-------|--------|
| 0 | - | 6 | Complement horizon |
| 1 | 0xA8 | 5 | Bulk |
| 2 | 0xA9 | 0 | Equality horizon |
| 3 | 0x28 | 5 | Bulk |
| 4 | 0x29 | 6 | Complement horizon |

Table 12. Shell trace for one four-byte full word F (one canonical turn).

The measured shell sequence is [6, 5, 0, 5, 6]. A second turn repeats this pattern over eight bytes at modal depth 16, producing nine shell readings. `docs/Findings/Analysis_hQVM_Wavefunction.md` reports [0, 1, 6, 1, 0] under arch_shell = 6 - k. The converted popcount sequence is [6, 5, 0, 5, 6]. Both charts start from the complement horizon and agree on symmetry about the equality transit and on passage through bulk shell 5 only.

F(m)^2(rest) = rest for all 64 micro-references (Theorem T6). After one F(m) from rest, all 64 images remain on shell 6. The Z2 carrier flip acts within the complement horizon at the byte-word level.

### 4.11 Probe-Word Reachability

Structured probe words from `docs/Findings/Analysis_hQVM_Wavefunction.md` Section 13 were tested on the word graph (all 64 micro-references per probe type):

| Probe | Reach | Full Omega | On shell 0 |
|-------|------:|------------|------------|
| canonical F | 2 | No | 0 |
| canonical 4-fam W2 | 128 | No | 64 |
| canonical F^2 (depth-8) | 1 | No | 0 |
| same-fam 00 | 1 | No | 0 |
| same-fam 11 | 1 | No | 0 |
| reverse fam order | 2 | No | 0 |

Table 13. Probe-word reachability from rest.

Only the canonical 4-family W2 word reaches both horizons. Same-family and reverse-order probes collapse to trivial reachability, confirming that the full K4 family cycle is required for horizon transport at the word level. Edge-mode sweeps varying the operator set per included micro-reference (W2+W2' only, F only, or all three) give P(full) = 0 at all tested p in {0.05, 0.10, ..., 1.0}. F edges confine reachability to the 128-state horizon orbit at every tested p.

Flat micro-references give a parallel result. A micro-reference m is flat when all four family bytes have zero fold disagreement. The kernel has zero flat micro-references, as all 64 are curved under the four-family palindromic criterion. Word percolation with a flat-only pool yields P(pair) = 0 at all tested p. The curved pool (all 64 micro-references) yields P(pair) = 1.0 and Reach = 128 for any nonempty selection at p at least 0.1.

## 5. The Hierarchy of Root Completion

### 5.1 Root Completion as a Hierarchy

The Square-Root Cluster Theorem establishes that cluster size is controlled by the transport rank r(A), with full connectivity requiring r = 6 and odd-parity transport. A generator set can reach full connectivity while its curvature spectrum, channel statistics, and holonomy transport remain incomplete. The reason is that the root can be full-dimensional while remaining sparsely populated, anisotropically branched, or unconposed into closure operators.

Five distinct coverage observables, each governed by the transport root, turn on at different generator fractions p because each requires a stronger condition on root(A). The conditions form a strict hierarchy. Each is a stronger condition than the previous one on root(A). The reported p_c values are 50% onset estimates for these Boolean events on the fixed 4096-state kernel (Appendix B).

1. **Orbit reachability.** The root is full-dimensional (r = 6 with odd parity).
2. **Defect spectrum.** The root is full-dimensional and uniformly covered by all seven transport weights.
3. **Channel isotropy.** The root produces symmetric branching (H1 approaches 7).
4. **Two-step uniformization.** The root is saturated at depth 2, yielding uniform two-step multiplicity.
5. **Holonomy transport.** The root is composed into depth-4 closure operators, activating full word availability.

Each observable imposes a stronger condition on the root than the previous one. The weak-threshold ordering gate reports p_c(span) approximately 0.0221, p_c(full) approximately 0.0297, p_c(spectrum) approximately 0.0398, p_c(h1_wk) approximately 0.0647, and p_c(word_evt) approximately 0.3379 at 50% onset with n = 200 (Appendix B.2). Event E_spectrum in the byte-fraction curvature sweep (Table 15) crosses at p_c approximately 0.0544 with E[#bytes] approximately 13.9 (part 2, Section 5).

### 5.2 The Five Critical Scales

| Event or observable | Protocol | p_c | E[active groups] at onset |
|------------------|----------|-----|---------------------------|
| E_span | byte fraction | ~0.022 | ~5.5 bytes |
| E_full | byte fraction | ~0.029 | ~7.4 bytes |
| E_full | family fraction | ~0.15 | ~0.6 families |
| E_full | q6-class fraction | ~0.105 | ~6.7 classes |
| E_full | micro-ref fraction | ~0.091 | ~5.8 refs |
| E_pair | word micro-ref fraction | ~0.103 | ~6.6 refs |
| E_spectrum | byte fraction | ~0.054 | ~14 bytes |
| Two-step cover (mean E[cover]) | byte fraction | ~0.7 | ~179 bytes |
| Two-step saturation (unif_st) | byte fraction | ~0.85 | - |
| Holonomy tau_cycle/Delta | byte fraction | - | governed by 1 - (1 - p^4)^64 |

Table 14. Critical thresholds under generator restriction.

State-space accessibility turns on near p approximately 0.03. Curvature-spectrum completion and future-cone entropy saturation require substantially larger generator sets (Table 14). At p_c(full) approximately 0.029 the bulk is reachable while spectrum completion waits until p_c approximately 0.054, channel saturation until larger p, and holonomy transport until word availability crosses the holonomy transport threshold. Kernel transport coefficients stabilize after closure-word availability crosses that holonomy threshold.

### 5.3 Root Coverage Depth

Three observables measure how completely the transport root is populated beyond bare connectivity (E_full at p_c approximately 0.029). Each requires greater generator diversity on the same rank-6 root.

**Defect spectrum.** Transport defect weight between bytes b_i and b_j is d(b_i, b_j) = popcount(q6(b_i) XOR q6(b_j)). Event E_spectrum requires all seven weights among pairwise defects of the allowed byte set.

| p | E[#bytes] | P(spectrum complete) | E[#weights] |
|---|-----------|---------------------|-------------|
| 0.03 | 7.7 | 0.070 | 5.04 |
| 0.05 | 12.8 | 0.420 | 6.24 |
| 0.08 | 20.5 | 0.830 | 6.82 |
| 0.10 | 25.6 | 0.955 | 6.95 |
| 0.13 | 33.3 | 1.000 | 7.00 |

Table 15. Curvature spectrum completion versus byte fraction. p_c approximately 0.0544 (about 13.9 expected bytes, part 2 Section 5).

**Channel isotropy.** From rest, H1(p) is the Shannon entropy of the one-step next-state distribution when a byte is chosen uniformly from the allowed generator set. The full generator set gives H1 = 7 (Features 77 through 79 in `docs/reports/hQVM_Features_Report.md`).

| p | E[H1] | E[\|R1\|] |
|---|-------|-----------|
| 0.020 | 2.19 | 5.1 |
| 0.029 | ~2.5 | ~7 |
| 0.050 | 3.56 | 12.4 |
| 0.100 | 4.57 | 24.5 |
| 0.200 | 5.47 | 46.1 |
| 0.500 | 6.51 | 96.7 |
| 1.000 | 7.00 | 128.0 |

Table 16. Future-cone entropy versus byte fraction (n = 200). The p = 0.029 row is interpolated at the strong full-Omega threshold.

**Two-step uniformization.** From rest, the full generator set produces uniform two-step multiplicity, in which every state is reached by 16 of the 65536 ordered two-byte words (Feature 80).

| p | E[cover] | E[std] |
|---|----------|--------|
| 0.100 | 0.13 | 0.40 |
| 0.300 | 0.70 | 1.20 |
| 0.500 | 0.97 | 2.10 |
| 0.700 | 1.00 | 2.66 |
| 1.000 | 1.00 | 0.00 |

Table 17. Two-step multiplicity degradation (n = 150). E[cover] is the mean fraction of Omega states that receive at least one two-step image from rest.

These three observables form a strict depth hierarchy: spectrum completion (p_c approximately 0.054) precedes channel isotropy (H1 approaches 7 at larger p) precedes two-step uniformization (E[cover] near p = 0.7). A stronger per-sample uniformization criterion (all states covered with minimum multiplicity at least 15) crosses 50% of samples near p = 0.85.

### 5.4 Holonomy Transport

The holonomy-transport event is the availability of depth-4 closure operators. The relevant observable is micro-reference coverage by the eight-byte Z2 holonomy cycle, formed from two four-byte F words (`docs/Findings/Analysis_hQVM_Wavefunction.md`, Theorem T6). Under independent byte inclusion with probability p, four-byte F-word availability tracks the analytic formula:

```
E[micro_cov] = 1 - (1 - p^4)^64
```

E[micro_cov] is the expected fraction of the 64 micro-references covered by at least one available four-byte F word. Each micro-reference requires four independent byte inclusions, hence the factor p^4.

At p = 0.10, measured coverage is 0.0001 and tau_cycle/Delta remains near zero. At p = 0.50, coverage is 0.0645 and tau_cycle/Delta is 0.984. At p = 1.00, coverage is 1.000 and tau_cycle/Delta equals 7591/7392. Holonomy transport turns on only when word availability crosses this threshold. It is the condition that the root is composed into closure operators, the strongest condition in the hierarchy.

At the complete alphabet the holonomy-path coefficient satisfies

```
tau_cycle / Delta = 7591 / 7392 = 4 * sum_k C(6,k)^3 / (64 * sum_k C(6,k)^2)
```

The sum runs over shell popcount k. The denominator is the Vandermonde identity C(12,6) = 924.

Word percolation under the canonical operators confirms confinement directly. Each included micro-reference m brings both W2(m) and W2'(m) (gauge-complete per m):

| p | E[#m] | P(pair) | P(full) | E[R] |
|---|-------|---------|---------|------|
| 0.04 | 2.56 | 0.022 | 0.000 | 22 |
| 0.08 | 5.12 | 0.338 | 0.000 | 67 |
| 0.12 | 7.68 | 0.625 | 0.000 | 96 |
| 0.16 | 10.24 | 0.890 | 0.000 | 120 |
| 0.20 | 12.80 | 0.980 | 0.000 | 127 |
| 0.24 | 15.36 | 1.000 | 0.000 | 128 |
| 0.50 | 32.00 | 1.000 | 0.000 | 128 |

Table 18. Canonical-word percolation (conditional on |M| > 0 for threshold estimates).

The weak event E_hit_eq_word has p_c approximately 0.005 (E[#m] approximately 0.32). Complete pole pairing (E_pair) has p_c approximately 0.103 (E[#m] approximately 6.6). P(full) = 0 at all p including p = 1.0. Random subset search found no set of 16 or fewer micro-references that achieves full Omega under word dynamics.

### 5.5 Aperture Threshold Comparison

The aperture Delta generalizes across the hQVM(d) family. The mean byte-level fold disagreement, normalized by the number of palindromic phase pairs (Section 2.5), equals 1/2 exactly at every d from 1 through 8. The corresponding depth-four aperture follows directly:

```
Delta(d) = 1/(8d)
```

so that 8d times Delta(d) equals 1 exactly at every d. At d = 6 this gives Delta(6) = 1/48 = 0.020833, close to but distinct from the continuum CGM aperture Delta = 0.020700 used in the comparison below; the two differ by about 6 * 10^-4, a resolution-scale distinction already discussed in the wavefunction analysis (Analysis_hQVM_Wavefunction.md, Sections 16.5 through 16.7) and not reopened here.

CGM dimensionless constants were compared to empirically determined thresholds. The aperture gap Delta = 0.0207 is the residual informational aperture after depth-4 spinorial closure of byte-level fold disagreements (`docs/Findings/Analysis_hQVM_Wavefunction.md`, Sections 16.5 through 16.7). It equals 1 - rho to leading order, with rho the closure ratio, and matches the holonomic ratio delta_BU / m_a from `docs/Findings/Analysis_Monodromy.md`.

| Constant | Value |
|----------|-------|
| Delta (aperture gap) | 0.02070 |
| 1/48 (geometric quantization) | 0.02083 |
| 5/256 (dyadic approximant) | 0.01953 |
| m_a (observational aperture) | 0.19947 |

Table 19. CGM constants for threshold comparison.

| Event | p_c | E[#bytes] at p_c | p_c / Delta |
|-------|-----|------------------|-------------|
| E_span (weak) | 0.022 | ~5.5 | 1.04 |
| E_full (strong) | 0.029 | ~7.4 | 1.39 |

Table 20. Weak and strong byte thresholds versus Delta. Full threshold index: Table 14.

The weak threshold is closer to Delta (ratio approximately 1.04) than the strong threshold (ratio approximately 1.39). In the byte-fraction protocol, p_c(span) and Delta are the same order of magnitude. Delta is the depth-4 closure invariant; p_c(span) is its percolation onset under independent byte inclusion. The ratio p_c/Delta approximately 1.04 links horizon spanning to aperture scale.

The strong threshold is governed by the combinatorial requirement to include enough Q6 diversity to cover all seven shells, typically requiring at least one Q6 weight-6 byte or an equivalent combination of lower-weight bytes. Equivalently, E_full requires r(A) = 6 with odd-shell access.

Family-fraction (p_c approximately 0.15) and register-protocol thresholds (Q6-class p_c approximately 0.105, micro-reference p_c approximately 0.091) follow coupon-collector and exact-rank scaling respectively. Delta and m_a are the closure limits of the same hierarchy: coupon-collector thresholds are the group-sampling expression of rank coverage at finite d, and Delta = 1/(8d) is the spinorial residual at depth-4 saturation.

### 5.6 Critical-Size Fold-Triple Restriction

To probe the link between aperture-scale closure and horizon spanning, generator sets of fixed size |A| = k with k in {5,...,10} are constructed by mixing bytes from pools classified by fold-triple activity. Pool[c] contains bytes active on c of the three BU-fold boundaries. Two sampling regimes are compared. In the unrestricted regime, bytes are drawn from all 256 operators. In the chain-constrained regime, bytes are drawn only from the 64-byte minimal five-boundary mask.

Under unrestricted sampling, constitutional spanning P(span) remains above 0.93 for all tested fold-porosity targets when |A| is at most 10. At |A| = 5 with zero fold porosity, P(span) = 0.980. Spanning at small generator set size is cheap when bytes may be drawn from anywhere in the full operator set.

Under chain-constrained sampling at |A| = 5, P(span) = 0 until the realized fold porosity reaches approximately 0.20. Half-spanning occurs near pi_fold approximately 0.10, a factor of about 4.8 above Delta. For |A| at least 6 in the same pool, P(span) = 0 at all tested targets up to pi_fold = 0.20. Constitutional spanning in the minimal mask therefore requires at least one byte active on the fold triple. The onset porosity lies well above the aperture gap.

The weak byte-fraction threshold p_c approximately 0.022 for E_span tracks aggregate generator diversity and future-cone entropy growth. Delta = 1/(8d) is the spinorial closure limit of the same fold-disagreement field. At small |A| the onset porosity exceeds Delta because fewer bytes enter depth-4 averaging; both converge to aperture scale as restriction saturates.

## 6. Structural Completeness Under Restriction

The structural layer completes the CGM-native percolation observables under generator restriction, covering commutation structure, the U x V coset factorization, shell composition of reachable sets, permutation-class percolation, geometric porosity, plaquette defects, shell transition spectra, and the counting identity under partial restriction. Monte Carlo sample count n = 150 through 200 per sweep unless noted. Rows at p = 1.0 are the full-generator-set limit and match kernel invariants reported in `docs/reports/hQVM_Features_Report.md`.

### 6.1 Commutation and Q-Diversity

Bytes x and y commute on Omega if and only if q6(x) = q6(y) (Features 68 through 76 in `docs/reports/hQVM_Features_Report.md`). Exhaustive enumeration yields 1024 commuting ordered pairs out of 65536 (rate 1/64).

Each Q6 transport class contains exactly four bytes, one per family. Deterministic reachability from rest with a single q-class alphabet never exceeds 8 states, and exhaustive census over all 64 transport classes confirms that maximum. Monte Carlo at p at least 0.1 gives P(full) = 0 for single-q-class subsets and P(full) = 1 for independent byte inclusion. Q-diversity across transport classes fills the transport root, matching the rank condition r(A) = 6.

### 6.2 U x V Coset Factorization and Rectangularity

Omega has product structure Omega = U x V with |U| = |V| = 64 C64 cosets (`docs/reports/hQVM_Features_Report.md`). Under random byte subsets from rest, both marginals saturate by p = 0.10, where E[|U_hit|] = E[|V_hit|] = 64 and P(full) = 1. At p = 0.05, E[|U_hit|] approximately 63 and P(full) approximately 0.94. Full-Omega reachability implies simultaneous saturation of both coset factors. Partial marginals at low p track the bulk percolation transition.

Beyond marginal coverage, define rectangularity

```
rect(A) = |R| / (|U_R| |V_R|)
```

R is the reachable set from rest under A. |U_R| and |V_R| are the cardinalities of its projections onto the marginal factors U and V. Rectangularity equals 1 when R is a full coset rectangle U_R x V_R.

| p | E[rect] | E[\|R\|] | E[\|U\|] | E[\|V\|] |
|---|---------|----------|----------|----------|
| 0.020 | 0.629 | 1356 | 37.0 | 37.0 |
| 0.025 | 0.721 | 2030 | 46.9 | 46.9 |
| 0.030 | 0.856 | 2711 | 51.8 | 51.8 |
| 0.050 | 0.984 | 3922 | 62.6 | 62.6 |
| 0.100 | 1.000 | 4096 | 64.0 | 64.0 |
| 1.000 | 1.000 | 4096 | 64.0 | 64.0 |

Table 21. Rectangularity versus byte fraction (n = 200). Full Omega implies rect = 1.0.

The square-root theorem predicts rect = 1 for fiber-complete restrictions. The sweep shows the product form emerging with p and holding once full Omega is reached. The skew at low p is the signature of fiber-incomplete selections, where the U and V factors are reached asymmetrically.

### 6.3 Shell Enrichment and Mean Entanglement

For reachable sets R from rest, shell enrichment E(k) is the ratio of the fraction of R in shell k to the Omega baseline C(6,k)/64. Mean S(chi) = popcount(chi) over R equals 3.0 (the Omega mean) for all tested p at least 0.1 with enrichment 1.0 in every shell. This mean equals d/2 exactly at every d in the hQVM(d) family; the d = 6 value of 3.0 is the physical instance. Once the giant component forms, the reachable set preserves the binomial shell distribution even when |R| is below 4096. Below the giant threshold, enrichment departs from unity while mean S remains near 3.0. The compression from 50% byte-level fold disagreement at level 0 to the 2.07% structural aperture Delta is the percolation signature of depth-4 spinorial averaging over the reachable set.

### 6.4 Permutation-Class Percolation

The 256 bytes collapse to 128 distinct Omega permutations (shadow pairs b and b XOR 0xFE). Percolation sweeps treat each class as one group (one representative or both shadow bytes):

| Mode | p_c(full) | E[#classes] at p_c |
|------|-----------|-------------------|
| one_repr | 0.059 | ~7.6 |
| both_shadow | 0.058 | ~7.4 |

Table 22. Permutation-class thresholds (n = 200).

The permutation-class threshold is about twice the byte threshold p_c approximately 0.029, matching coupon-collector scaling on 128 groups versus 256. Shadow-pair inclusion shifts the threshold by less than 0.006.

### 6.5 Geometric Porosity

Byte fraction p is the experimental control for generator inclusion in Monte Carlo sweeps. The connection 1-form chain provides boundary-channel porosity:

```
pi_j(A) = (1/|A|) #{ b in A : ||A_j(b)|| > 0 }
```

pi_j(A) is the fraction of bytes in A with nonzero connection magnitude at phase boundary j. A_j(b) is the connection 1-form component at boundary j for byte b. The index j runs over the seven constitutional phase boundaries. The BU-fold porosity pi_BU uses the BU|BU boundary (bit 3 through 4). Fold porosity pi_fold averages ONA|BU, BU|BU, and BU|ONA.

Stratified sweeps vary pi_BU by mixing BU-active and BU-inactive byte pools while holding the remaining generator set structure fixed. Under this protocol, constitutional spanning P(span) remains near unity for pi_BU as low as 0.01. Full-Omega reachability P(full) rises only when pi_BU exceeds approximately 0.03. The weak spanning threshold on byte fraction p_c approximately 0.022 tracks aggregate generator diversity, not closure of the BU|BU channel alone.

Exhaustive search over all 2^7 boundary masks confirms that full Omega requires at least five active boundaries (Table 4). The minimal mask {CS|UNA, ONA|BU, BU|BU, BU|ONA, ONA|UNA} achieves full Omega with 64 bytes. No single boundary exclusion collapses reachability below 256 states except the four BU-fold channels.

At full generator set, every boundary has porosity pi_j = 0.75 and blockage fraction 1 - pi_j = 0.25 under the connection 1-form definition above.

Delta = 0.0207 indexes aggregate restriction of the BU-fold triple {ONA|BU, BU|BU, BU|ONA} under simultaneous closure. The weak spanning event E_span at byte fraction p_c approximately 0.022 is the percolation onset of the same BU-fold geometry.

### 6.6 Plaquette Loop Defect

A **plaquette** is an ordered pair of byte generators (x, y). The **plaquette defect** is

```
d(x,y) = popcount( q6(x) XOR q6(y) )
```

For an ordered byte pair (x, y), d(x,y) is the Hamming weight of the transport difference q6(x) XOR q6(y) in GF(2)^6.

The displacement invariant D = sum d / (2|Omega|) over all 256^2 byte pairs. Exhaustive evaluation gives D = 24.

Under generator restriction A:

```
D(A) = (1 / (2|Omega|)) sum_{x,y in A} popcount(q6(x) XOR q6(y))
```

D(A) averages the plaquette defect over all ordered pairs in the allowed generator set A, normalized by 2|Omega|. At full generator set, D(A) = 24 and the mean defect is 3.0 per plaquette. D(A) rises with |A| and saturates at 24 by |A| approximately 64 through 128 in random subsets. At |A| = 64, E[mean d] = 2.97. At |A| = 256, E[mean d] = 3.00. This is the percolation-layer enumeration of plaquette displacement D = 24, in the Regge calculus sense of deficit angles on lattice hinges (Regge, 1961).

The plaquette defect is the discrete curvature on the transport root. Its saturation at D = 24 is the large-generator-set limit in which the root is fully and uniformly covered, the same condition as defect-spectrum completion.

### 6.7 Spanning Transmission and Bulk Penetration

Define transmission and bulk penetration from rest as

```
T(A) = P(E_span)
-ln T(A)   = logarithmic failure rate of horizon-to-horizon bridging
bulk fraction = (count of bulk shells {1,...,5} in Reach) / 5
```

T(A) is the spanning transmission probability from rest. Reach denotes the reachable set under A. The bulk fraction counts how many of the five bulk shells appear in Reach.

| p | P(span) | -ln T | E[bulk shells hit]/5 |
|---|---------|-------|----------------------|
| 0.010 | ~0.14 | ~2.0 | ~0.20 |
| 0.022 | ~0.42 | ~0.87 | ~0.45 |
| 0.030 | ~0.75 | ~0.29 | ~0.70 |
| 0.050 | ~0.98 | ~0.02 | ~0.95 |

Table 23. Spanning transmission versus byte fraction (n = 300). Bulk fraction is the mean count of bulk shells {1,...,5} in the reachable set, divided by 5.

Partial generator sets block bulk-shell transit before horizons disconnect. At p = 0.020, T approximately 0.44, -ln T approximately 0.81, and bulk penetration is approximately 0.45.

### 6.8 Shell-Resolved Connectivity versus Topological Depth

Define topological depth and inward connectivity from the complement horizon anchor as

```
psi_topo(k) = 6 - k
C(k) = fraction of edges from shell-k states in R that reach shell k-1
```

psi_topo(k) measures inward distance from the complement horizon (k = 6 gives psi = 0). C(k) is the inward connectivity at shell k within reachable set R. Shell index k is the popcount shell label used throughout this document. At full generator set (p = 1.0), inward connectivity from the UV anchor (shell 6 to shell 5) is C = 0.094. Connectivity peaks at the equatorial transition shell 4 to shell 3 with C = 0.312. Shells 5, 4, 2, and 1 carry C between 0.016 and 0.234. Shell 0 has C = 0 because no inward shell exists below the equality horizon.

Below the giant threshold, C collapses first at intermediate psi (bulk equator), while the complement horizon retains edges from rest. Under full generator set, inward connectivity is lowest at the UV anchor and peaks at the equatorial transition. Under restricted generator set, intermediate depths lose connectivity before the horizon anchor does.

### 6.9 Shell Transition Operators

Empirical shell transition matrices M_A(q) are built from uniform byte steps in the allowed generator set, restricted to fixed q6 weight q. For odd q, Tr(M_q) = 0 by parity. The carrier trace C(q) = Tr(M_q^2) matches the closed-form table of `docs/Findings/Analysis_Compact_Geometry.md`, Section 2, at full generator set:

| q | C(q) |
|---|------|
| 0 | 7 |
| 1 | 28/9 |
| 2 | 7/3 |
| 3 | 52/25 = 2.08 |
| 4 | 7/5 |
| 5 | 28/9 |
| 6 | 1 |

Table 24. Carrier traces C(q) = Tr(M_q^2) at full generator set.

The binomial shell moment is a population invariant of the shell chart:

```
M_shell = sum_k k C(6,k) = 192
```

M_shell is the first moment of the binomial shell chart. C(6,k) is the binomial coefficient counting six-bit strings of popcount k.

Under generator restriction, C(q) remains stable for each q-class present in A while the reachable shell support shrinks. For example, C(3) = 2.08 is unchanged from p = 0.05 through p = 1.00 even as Tr(M_3) vanishes when weight-3 bytes are absent. The carrier trace is a property of the transport root that is invariant under how much of the root is reached.

### 6.10 Counting Identity Under Partial Restriction

The square-root theorem predicts that the counting ratio |H_comp,r|^2 / |Omega_r| persists under partial generator access whenever the reachable set remains a product in U x V. Under fiber-complete restriction, |Reach| = |U_R| |V_R| and the ratio equals unity by construction. This section tests that persistence under byte-fraction restriction.

Full Omega satisfies |H|^2 = |Omega| with |H| = 64 and |Omega| = 4096. The test uses n = 100 samples per p.

| p | E[Omega_r] | E[H_comp,r] | E[H_comp,r^2] | Ratio |
|---|-----------|-------------|---------------|-------|
| 0.05 | 3937 | 64 | 4096 | 1.040 |
| 0.10 | 4096 | 64 | 4096 | 1.000 |
| 0.20 | 4096 | 64 | 4096 | 1.000 |
| 0.50 | 4096 | 64 | 4096 | 1.000 |
| 1.00 | 4096 | 64 | 4096 | 1.000 |

Table 25. Counting ratio under byte-fraction percolation.

The complement horizon is fully reached (64 states) for p at least 0.05, before the full 4096-state space saturates at higher p. At p = 0.05 the mean reachable set size E[Omega_r] is 3937 while E[H_comp,r] is already 64, giving ratio 1.040. The ratio is 1.000 for p at least 0.10. The holographic identity therefore survives partial generator access once the root is fully recovered, even when the square remains partially populated.

## 7. Conclusions

Generator-restricted percolation on the CGM kernel closes algebraically on transport rank. Ancestry preservation forces a conjugate product structure, so under fiber-complete restriction cluster geometry, coverage fractions, and register-protocol thresholds are determined by the same transport object. For r(A) at least 1,

```
log2|Reach(A)| = 2 r(A),    |Reach(A)| = (2^r(A))^2,    |Reach(A)| / |Omega_d| = (2^r(A) / 2^d)^2
```

At r(A) = 0, root(A) = 2 and |Reach(A)| = 2. The observable cluster is the square of its transport root throughout the hQVM(d) family.

**Square-root state space.** Omega is a holographic product U x V with |U| = |V| = 64 at d = 6. The square root is the coordinate form of observables on a product state space forced by ancestry preservation.

**The two regimes.** A single byte is a half-cycle acting on the full product; a four-byte canonical word composes two half-cycles into a closed invariant within the root at shells 0 and 6.

**Two register protocols.** Restriction by transport value spans GF(2)^d directly. Restriction by payload reduces to a rank problem on a quotient of dimension d - 1, because the transport map pairs each payload with its complement under the all-ones vector epsilon_d. Both protocols obey Theorem 3.1; the micro-reference protocol admits closed-form rank thermodynamics (Section 4.3.5).

**Root completion hierarchy.** Five coverage observables under byte-fraction restriction turn on at distinct p, from full transport rank through defect-spectrum completion, channel isotropy, two-step uniformization, and holonomy transport. Bulk reachability at p_c(full) approximately 0.029 precedes spectrum completion at p_c approximately 0.054 and holonomy transport governed by 1 - (1 - p^4)^64.

**Counting and stabilizers.** The identity |H|^2 = |Omega| survives partial generator access above a byte-fraction threshold. Horizon stabilizers vanish under random byte inclusion.

**Geometry and aperture.** Constitutional spanning lies near the aperture scale Delta. Delta(d) = 1/(8d) across the hQVM(d) family.

**Kernel census saturation.** Plaquette normalization D(A) saturates at D = 24, holonomy transport at tau_cycle/Delta = 7591/7392, and inward connectivity C(k) reaches the full-alphabet profile at p = 1.

## Appendix A. Benchmark Specification for Mechanistic Interpretability

This appendix specifies four supervised task families built on the percolation system for future studies that train models on data with known generative structure and test whether analysis tools recover that structure from model internals. Each task states the input encoding, the exact label function, the control that separates mechanism from correlate, and the failure mode the task is designed to expose. All labels are computed by enumeration. No label is statistical.

The tasks target readers who want ground-truth algebra independent of any trained model. They do not assume that linear probes, sparse autoencoders, causal tracing, or activation patching will recover transport rank or the true dynamics map; they supply exact labels and paired controls under which such recovery can be scored. Training models and reporting interpretability outcomes remain outside the scope of the percolation census in the main text.

### A.1 Task 1: Rank Recovery Under Size-Controlled Evaluation

**Input.** A 256-bit membership vector encoding an allowed generator set A.

**Label.** The transport rank r(A) = rank_GF(2)(span{q6(b) : b in A}), or the reachable set size |Reach(A)|.

**Evaluation design.** Two evaluation strata are used. The first stratum samples generator sets by independent byte inclusion. The second stratum fixes generator-set size while restricting transport rank. One example is the even-weight transport subspace, which permits large generator sets while capping reachability at 1024 states. A second example selects small generator sets with transport rank 6. This evaluation design separates transport-rank dependence from generator-set size dependence.

**What internals should show.** A model that generalizes to the rank-restricted stratum must represent the q6 span, a six-dimensional linear structure over GF(2), somewhere in its activations. A probing method succeeds on this task when it locates that subspace representation. Rank is a matroid invariant of the q6 span and is not a linear function of the raw 256-bit membership vector, so linear probes on the input embedding alone cannot recover r(A) without learning an intermediate representation.

**Difficulty dial.** The gap between the correlated and rank-restricted strata is continuous, because mixing fraction lambda of rank-restricted examples into training tunes how strongly the shortcut is penalized, from lambda = 0 (shortcut viable) to lambda large (rank computation forced).

**Sparse feature alignment (open).** When sparse autoencoders or similar methods are applied to models trained on this task, the generative process defines exact candidates at several levels (individual generators, transport directions, span basis, rank). Which level, if any, appears in learned features is an evaluation question, not a property of the kernel.

### A.2 Task 2: Mechanism Versus Correlate via the Dynamics Shuffle

**Input.** A set of micro-references M in {0,...,63} (64-bit membership vector), plus a start state, under one of two data-generating conditions.

**Condition T (true dynamics).** Trajectories generated by the true word operators W2(m), W2'(m) for m in M. From rest, every trajectory remains on the 128 horizon states.

**Condition S (shuffled dynamics).** Identical labels and set statistics, but each byte's transition table is replaced by a random other byte's table. Trajectories leak into the bulk, and P(full) reaches 1 by p = 0.30.

**Label.** Reachability of a query state from the start state under the given micro-reference set.

**What this isolates.** Marginal input statistics are identical between conditions by construction. Any representation difference between a model trained on T and a model trained on S is attributable to the label-to-dynamics map. This gives a matched-pair benchmark for whether an interpretability tool detects the generative mechanism, since the tool should report a two-cluster horizon structure (a union of 64-node matchings) for the T-model and a diffuse reachability structure for the S-model. The same paired inputs support comparing probes, sparse autoencoders, or causal methods under matched statistics; whether those methods diverge between T and S is left to experiment.

### A.3 Task 3: Which Threshold Does a Probe See

**Input.** A generator set membership vector, sampled at byte fractions spanning p = 0.01 to 1.0.

**Labels (five, on the same inputs).** E_full (onset p approximately 0.029), E_spectrum (p approximately 0.054), H1 within tolerance of 7 (larger p), two-step uniformity (mean E[cover] near p approximately 0.7, saturation unif_st near p approximately 0.85), and holonomy-word coverage micro_cov (governed by 1 - (1 - p^4)^64).

**Purpose.** These five events have distinct coverage onsets on identical inputs (Table 14). Training one model per label and applying the same analysis tool to all five tests whether the tool distinguishes which coverage criterion a model encodes. The fourth and fifth labels use higher p_c, because uniformization depends on path multiplicities across the allowed alphabet, while holonomy coverage depends on four-byte word completion, a conjunctive statistic over single-byte inclusion.

### A.4 Task 4: Anchor-Dependent Structure

**Input.** A start state (24-bit or shell-labeled) and a micro-reference set.

**Label.** The shell profile of the word-reachable set. From horizon anchors this is {0, 6}. From a shell-k bulk anchor it is {k, 6-k}.

**Purpose.** The correct internal model is a single global rule (shell s maps to 6 - s) that produces different observable confinement depending on the anchor. A model may instead memorize per-anchor lookup behavior. The two hypotheses agree on all training anchors and separate on held-out shells. An internals analysis should find one shared reflection circuit governing all shells. Whether this pattern of anchor-dependent confinement offers a useful structural comparison for context-dependent behavior in trained systems is left for the benchmarks in this appendix to test.

### A.5 Scope

The four tasks above specify benchmarks for recovery of CGM generative structure, namely GF(2) rank, mechanism versus correlate, separable coverage observables, and anchor-dependent rules. The percolation study establishes the kernel census and scaling relations under generator restriction, including large-generator-set saturation to kernel invariants, transport-rank scaling, coupon-collector thresholds across restriction protocols, and the hierarchy of coverage thresholds. Dataset generation for all four tasks requires only the transition table and breadth-first search (Appendix B).

## Appendix B. Reproducibility and Supplementary Output

Scripts in `experiments/`: `hqvm_percolation_analysis_1.py` through `_5.py` (byte, word, structural, theorem gates, and hQVM(d) family scaling). Runner `_run.py` executes parts 1 through 4. Combined stdout: `hqvm_percolation_analysis_results.txt` (parts 1 through 4) and `hqvm_percolation_analysis_5_results.txt` (family scaling and exact rank machinery). Random seed 20260702. Monte Carlo n = 300 per sweep in parts 1 and 2 unless a table states otherwise, 150 through 200 in part 3. Part 4 and the family gates are deterministic. Transition-table precompute dominates wall time (about 2 minutes per fresh process). Parts 1 through 3 add tens of minutes at n = 300 on a single Python 3.14 thread.

The subsections below map supplementary tables in the results files to main-text content. Principal tables appear in the byte, word, and structural layers.

### B.1 Part 1 supplements

| Results section | Content |
|-----------------|---------|
| I.5, II.H | Family restrictions, horizon stabilizers |
| III.B, VIII | Byte-fraction sweep tables, threshold comparisons |

### B.2 Verification gates

27 deterministic PASS gates from part 4: holographic identity, fiber-complete product cluster (structured case table at d = 6, including r = 0 gauge doublet), parity obstruction, word confinement, criticality ordering (weak_threshold_ordering: span < full < spectrum < h1_wk < word_evt), holonomy scaling, plaquette census D = 24.

Family and register-protocol gates from part 5: square-root identity across d = 1 through 8 (52 of 52 PASS); rank-reachability equivalence under micro-reference restriction (exhaustive for d up to 4); exact rank distribution (brute force through d = 4, pair-quotient at d = 5, algebraic through d = 8); asymptotic threshold constant c_inf approximately 1.2665 fitted at d = 28 and 32; Delta(d) = 1/(8d) and mean entanglement d/2 at every d from 1 through 8. Source: `gyroscopic/hQVM/family.py`.

### B.3 Part 3 supplements

| Results section | Content |
|-----------------|---------|
| 1, 2, 4 | Commutation (§6.1), future-cone entropy (Table 16), shell enrichment (§6.3) |
| 5 | Two-step multiplicity (Table 17) |
| 6 | Fold disagreement (Table 2), phase-net (Table C.1) |
| 8 | Permutation-class thresholds (Table 22) |
| 11 | Spanning transmission (Table 23) |
| 12 | Shell-resolved connectivity (§6.8) |
| 13 | U x V rectangularity (Table 21) |
| 15 | Shell transition operators (Table 24) |

### B.4 Part 2 supplements

| Results section | Content |
|-----------------|---------|
| 4 | Canonical-word percolation (Table 18) |
| 5 | Curvature spectrum (Table 15) |
| 6 | Null model (Table 11) |
| 9 | Probe words (Table 13) |
| 12 | Shell trace (Table 12) |

## Appendix C. Supplementary Tables

### C.1 Phase-Net Vector Reachability

Each of the 16 phase-net vectors is associated with 16 bytes. None generates a giant component alone. All achieve constitutional spanning.

| Phase-net | Bytes | Reach | Giant |
|-----------|-------|-------|-------|
| (0,0,0,0) | 16 | 64 | No |
| (1,0,0,0) | 16 | 64 | No |
| (0,0,0,1) | 16 | 256 | No |
| (0,0,1,0) | 16 | 256 | No |
| (0,0,1,1) | 16 | 256 | No |
| (0,1,0,0) | 16 | 256 | No |
| (0,1,0,1) | 16 | 256 | No |
| (0,1,1,0) | 16 | 256 | No |
| (0,1,1,1) | 16 | 256 | No |
| (1,0,0,1) | 16 | 256 | No |
| (1,0,1,0) | 16 | 256 | No |
| (1,0,1,1) | 16 | 256 | No |
| (1,1,0,0) | 16 | 256 | No |
| (1,1,0,1) | 16 | 256 | No |
| (1,1,1,0) | 16 | 256 | No |
| (1,1,1,1) | 16 | 256 | No |

Table C.1. Reachability for all 16 phase-net vectors.

Vectors with only CS active ((1,0,0,0) and (0,0,0,0) up to family relabeling) reach 64 states. All other single vectors reach 256 states. Requiring all four phases active (16 bytes with phase-net (1,1,1,1)) yields 256 reachable states. Requiring at least three active phases (80 bytes) restores full Omega.

### C.2 Fold Disagreement Cumulative Unions

| Alphabet | Bytes | Reach | Shells | Span | Giant | Full |
|----------|-------|-------|--------|------|-------|------|
| Fold disagree <= 0 | 16 | 64 | (0,2,4,6) | Yes | No | No |
| Fold disagree <= 1 | 80 | 4096 | (0-6) | Yes | Yes | Yes |
| Fold disagree <= 2 | 176 | 4096 | (0-6) | Yes | Yes | Yes |
| Fold disagree <= 3 | 240 | 4096 | (0-6) | Yes | Yes | Yes |
| Fold disagree <= 4 | 256 | 4096 | (0-6) | Yes | Yes | Yes |

Table C.2. Cumulative fold-disagreement reachability.

## References

Alon, N., Benjamini, I., and Stacey, A. (2004). Percolation on finite graphs and isoperimetric inequalities. *Annals of Probability*, 32(2), 1727-1745.

Broadbent, S. R., and Hammersley, J. M. (1957). Percolation processes I. Crystals and mazes. *Mathematical Proceedings of the Cambridge Philosophical Society*, 53(3), 629-641.

Dixon, J. D. (1969). The probability of generating the symmetric and alternating groups. *American Mathematical Monthly*, 76(6), 689-691.

Fulman, J., and Goldstein, L. (2014). Stein's method and the rank distribution of random matrices over finite fields. *Annals of Probability*, 42(3), 975-1001.

MacWilliams, F. J., and Sloane, N. J. A. (1977). *The Theory of Error-Correcting Codes*. North-Holland.

McDiarmid, C. (1981). General percolation and random graphs. *Advances in Applied Probability*, 13(1), 40-60.

Regge, T. (1961). General relativity without coordinates. *Il Nuovo Cimento*, 19(3), 558-571.

Toth, B. (2008). Corner percolation on Z^2 and the square root of 17. *Annals of Probability*, 36(5), 1708-1732.

Ungar, A. A. (2008). Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity. World Scientific.
