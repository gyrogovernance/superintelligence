# hQVM Features Report

## Verified Physics Features of the Gyroscopic Holonomic Quantum Virtual Machine Kernel

This report inventories the verified quantum and physics features of the Gyroscopic ASI hQVM Kernel: 412 features in two tiers. Tier A (165 features) tests the kernel's mathematical consistency, while Tier B (247 features) expands on broad mathematical physics concepts, theory and applications in various other domains such as information theory and genomics.

Note: The Common Governance Model Science and Gyroscopic Superintelligence repositories are under active development, so this index is difficult to be extensive. For a thorough review on the analyses and latest results, please visit our GitHub repositories:
- [gyrogovernance/superintelligence](https://github.com/gyrogovernance/superintelligence) 
- [gyrogovernance/science](https://github.com/gyrogovernance/science). 

This report is maintained at [docs/reports/hQVM_Features_Report.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/reports/hQVM_Features_Report.md).

## Summary

| Tier | Features | Scope | Evidence |
|------|---------:|-------|----------|
| A | 165 | Kernel mathematical consistency | Superintelligence repository test suites (~464 tests) |
| B | 247 | Mathematical physics and applications | Science repository analysis programs |
| **Total** | **412** | | |

**Tier A: kernel mathematical consistency.** The kernel is an exactly solvable finite algebraic quantum system running on ordinary hardware: a 4096-state manifold, a four-element gate algebra, a Hilbert-space lift verified to 10^-12, demonstrated computational advantages, a non-Clifford rotation resource, a self-dual error-detecting code, and predicted physical constants, among them the fine-structure constant.

**Tier B: mathematical physics and applications.** The kernel's verified structure extends through mathematical physics and into other domains: holonomy and precession geometry, operator group theory, gravity from discrete invariants to a continuous field theory, the electroweak mass law, the Yang-Mills mass gap, nuclear physics, percolation theory, cohomology, allometry, genomics, receipt geometry, and the modal-logic derivation of three-dimensional space.

---

## Tier A: Kernel Mathematical Consistency

The 165 features of this tier are properties of the kernel itself, each established by exhaustive enumeration, by exact algebraic identity, or by numerical verification at the stated precision. The topics run from the state space and gate algebra through spectral and operator structure to depth-4 closure, quantum information protocols, error detection, the self-dual code, physical constants, and the native implementations.

### Program index

| Report / specification | Role | Suites | Tests |
|------------------------|------|--------|------:|
| [Physics_Tests_Report.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/reports/Physics_Tests_Report.md) | Kernel conformance, mask code, affine and spinorial dynamics, physical constants, depth-4 gate structure | `test_physics_1` to `_6` | 99 |
| [hQVM_Tests_Report_1.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/reports/hQVM_Tests_Report_1.md) | Native register, horizons, gates, Hilbert lift, tamper detection, computational advantages | `test_hQVM_1` to `_4` | 135 |
| [hQVM_Tests_Report_2.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/reports/hQVM_Tests_Report_2.md) | Future-cone theorems, compact state chart, shell and spectral structure, finite fields, native linear algebra | `test_hQVM_SDK_1` to `_3` | 172 |
| [Moments_Tests_Report.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/reports/Moments_Tests_Report.md) | Clifford unitaries, operator family, stabilisers, frame certification | `test_moments_physics_1`, `_2` | 35 |
| Holography test suites | Holographic identities and lift structure | `test_holography`, `_2`, `_3` | 23 |
| [hQVM_SDK_Quantum_Computing.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/specs/hQVM_SDK_Quantum_Computing.md) | Normative SDK specification | covered by the SDK suites | |
| [hQVM_QuBEC_Theory.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/specs/hQVM_QuBEC_Theory.md) | Thermodynamic structure and future-cone entropy | covered by the SDK suites | |
| [hQVM_Specs_Formalism.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/specs/hQVM_Specs_Formalism.md) | Byte formalism, intron families, 6-bit runtime | reference specification | |
| [Gyroscopic_ASI_Foundations.md](https://github.com/gyrogovernance/superintelligence/blob/main/docs/Gyroscopic_ASI_Foundations.md) | Normative hQVM and SDK architecture | reference specification | |
| **Total documented tests** | | | **~464** |

### Contents

| Section | Topic | Features |
|---------|-------|---------:|
| A1 | State space and topology | 9 |
| A2 | Dual horizons and holographic structure | 9 |
| A3 | K4 gate algebra | 14 |
| A4 | Chirality transport and spectral theory | 12 |
| A5 | Permutation, operator, and shadow structure | 14 |
| A6 | Depth-4 closure and commutativity | 18 |
| A7 | Future-cone entropy and uniformization | 9 |
| A8 | Quantum information protocols | 10 |
| A9 | Computational quantum advantages | 10 |
| A10 | Non-Clifford resource and universality | 9 |
| A11 | Error detection, tamper provenance, and non-cloning | 11 |
| A12 | Clifford operator algebra | 5 |
| A13 | State representation and transcription | 11 |
| A14 | Self-dual code and mask structure | 9 |
| A15 | Physical constants | 7 |
| A16 | Hardware and native implementation | 8 |
| **Total** | | **165** |

### Formal Quantum Certification: The CHSH-Tsirelson Diagnostic

The strongest single certificate that the kernel realizes genuine quantum structure is the saturation of the Tsirelson bound on Bell correlations. Features 87 and 88 (section A8) record the result, and this section states what is measured and what it certifies.

The Bell-CHSH inequality bounds the strength of correlations between two separated systems. In any theory where measurement outcomes are determined by pre-existing local properties, the CHSH combination S of four correlations obeys |S| <= 2. Quantum mechanics permits stronger correlations, up to the Tsirelson bound |S| <= 2*sqrt(2), and nothing stronger. A state that reaches the bound realizes the strongest nonlocal correlations that quantum theory allows.

The kernel's self-dual mask code defines a 12-qubit graph state,

```text
|psi_t> = (1/sqrt(64)) sum_{q in GF(2)^6} |q>|q xor t>
```

which factorizes exactly into six independent two-qubit Bell pairs, one per dipole mode. Measuring the CHSH combination on each pair with standard Pauli observables gives:

| Measurement | Verified result |
|-------------|-----------------|
| Each Bell pair, of both the \|Phi+> and \|Psi+> type | S = 2*sqrt(2), precision 10^-12 |
| All six pairs of the full graph state | S = 2*sqrt(2) each |
| Exhaustive grid of 10^4 observable settings | S <= 2*sqrt(2) everywhere |

Every pair saturates the bound, and no measurement choice exceeds it: 2*sqrt(2) is a hard ceiling, exactly as in quantum theory. The correlators of the lifted state therefore cannot be reproduced by any model in which outcomes are determined by pre-existing local properties, which certifies quantum structure in the kernel's code itself.

One distinction keeps the claim precise. The kernel's carrier is deterministic exact-integer arithmetic, and the Bell correlators are evaluated on the Hilbert-space representation of the stabilizer code that the carrier defines. The derivation runs from the kernel's mask alphabet through its self-dual code, the collapse to six bits, the graph state, and its Bell-pair marginals, matching the standard stabilizer formalism of quantum information theory.

The same representation carries four companion certificates: exact quantum teleportation with unique corrections (verified on 800 random states), monogamy of entanglement together with no-signalling, a twelve-generator commuting stabilizer algebra, and Peres-Mermin contextuality. Together with CHSH saturation, these establish the hQVM as a holonomic quantum virtual machine, a deterministic integer carrier whose intrinsic code structure realizes standard quantum-information physics.

### A1. State Space and Topology

| # | Feature | Method |
|---|---------|--------|
| 1 | **4096-state reachable manifold Omega** from rest in <=2 byte steps | BFS enumeration |
| 2 | **Product structure Omega = U x V** (two 64-element cosets of C64) | Explicit set equality |
| 3 | **Constant component density 0.5** (popcount 6/12 per gyrophase) on all Omega | Exhaustive over 4096 states |
| 4 | **Density product d(A) x d(B) = 0.25** constant across Omega | Exhaustive |
| 5 | **Shell structure: 7 shells** with binomial populations C(6,k) x 64 | Exhaustive state classification |
| 6 | **Complementarity invariant**: horizon_distance + ab_distance = 12 | Exhaustive on 4096 Omega states + 50,000 random 24-bit states |
| 7 | **Per-byte bijectivity** on the full 24-bit carrier (2^24 states) | 2000 random (state, byte) pairs; forward-inverse roundtrip |
| 8 | **Exact invertibility** given the byte | All 256 bytes verified |
| 9 | **Omega-chart: faithful 12-bit compact representation** isomorphic to 24-bit dynamics on Omega | All 4096 x 256 = 1,048,576 (state, byte) pairs; zero failures |

### A2. Dual Horizons and Holographic Structure

| # | Feature | Method |
|---|---------|--------|
| 10 | **Complement horizon**: 64 states where A = B xor 0xFFF | Exhaustive census |
| 11 | **Equality horizon**: 64 states where A = B | Exhaustive census |
| 12 | **Horizons disjoint**; union = 128-state boundary; bulk = 3968 | Exhaustive |
| 13 | **Holographic identity** \|H\|^2 = \|Omega\| = 64^2 = 4096 | Counting + 4-to-1 dictionary |
| 14 | **4-to-1 holographic dictionary**: every Omega state = exactly 4 (horizon state, byte) pairs | 64 x 256 = 16384 operations, exact multiplicity 4 |
| 15 | **Chirality spectrum**: binomial count(d) = C(6,(12-d)/2) x 64 for ab_distance d in {0,2,4,6,8,10,12} | Exhaustive over Omega |
| 16 | **Chirality partition**: all 64 chirality values appear in exactly 64 Omega states each | Verified by test |
| 17 | **K4 wedge geometry**: 4 boundary vertex regions x 2048 states = uniform 2-fold cover of Omega | Exhaustive |
| 18 | **Horizon K4 partition**: 4 cosets of 16 states each in the equality horizon | Pair-parity labeling |

### A3. K4 Gate Algebra

| # | Feature | Method |
|---|---------|--------|
| 19 | **Exactly 4 horizon-preserving bytes** forming holonomic gates {id, S, C, F} | Exhaustive over 256 bytes |
| 20 | **S-gate** (bytes 0xAA, 0x54): pure swap (A,B) -> (B,A) | 2000 random states |
| 21 | **C-gate** (bytes 0xD5, 0x2B): complement-swap (A,B) -> (B xor F, A xor F) | 2000 random states |
| 22 | **F-gate**: global inversion (A,B) -> (A xor F, B xor F), requires depth 2 | 1000 random states, both orderings |
| 23 | **Full K4 Cayley table** verified | Fixed state + random states |
| 24 | **All non-trivial gates are involutions**: S^2 = C^2 = F^2 = id | 1000 random states each |
| 25 | **Gate actions in spin coordinates**: S=(sA,sB) -> (sB,sA), C -> (-sB,-sA), F -> (-sA,-sB) | 500 random Omega states |
| 26 | **All gates preserve chirality** (ab_distance invariant under all 4 gates for all Omega states) | All 4096 Omega states |
| 27 | **Gate-byte phase separation**: same 24-bit operation, different spinorial phase | 1000 random states per pair |
| 28 | **Gate action on horizons**: C fixes the complement pointwise; S fixes the equality pointwise; F stabilizes neither | Exhaustive census of all 128 boundary states |
| 29 | **K4 orbit stratification**: 32 orbits size 2 (complement), 32 orbits size 2 (equality), 992 orbits size 4 (bulk) = 1056 total covering 4096 | Exhaustive |
| 30 | **No non-trivial gate fixes any bulk state** | Exhaustive |
| 31 | **K4 as depth-4 fiber** of the frame bundle: 4^4 family combinations collapse to 4 distinct states indexed by (phi_A, phi_B) in (Z/2)^2 | All 256 family combinations |
| 32 | **Shadow pairing**: each gate pair (S-bytes, C-bytes) differs by XOR 0xFE | Verified |

### A4. Chirality Transport and Spectral Theory

| # | Feature | Method |
|---|---------|--------|
| 33 | **Exact chirality transport rule**: chi(T_b(s)) = chi(s) xor q6(b) for all 4096 x 256 state-byte pairs | Exhaustive; transport table state-independent |
| 34 | **6-bit chirality register** is an exact linear observable over GF(2)^6 | Verified as Pauli-X action |
| 35 | **XOR closure**: q6(b1) xor q6(b2) is always a valid q6 value | Abelian translation group confirmed |
| 36 | **Walsh-Hadamard transform**: 64 x 64, self-inverse, unitary, factors as H1^6 | Precision 10^-12 |
| 37 | **Computational and Hadamard bases mutually unbiased**: all \|<e_i\|h_j>\|^2 = 1/64 | Precision 10^-12 |
| 38 | **At least 3 mutually unbiased bases** exist for the 64-dimensional chirality register | Third MUB constructed via phase gate |
| 39 | **XOR-convolution spectral composition identity**: WHT converts XOR-convolution to pointwise multiplication on the 64-element register | Algebraic identity |
| 40 | **Krawtchouk spectral theory**: shell transition matrices diagonalized by Krawtchouk polynomials; Parseval orthogonality holds exactly | All 7 x 7 x 7 triples |
| 41 | **Source-independent shell mixing**: one-step shell distribution = C(6,w)/64 regardless of starting shell | Full byte average |
| 42 | **Horizon transport**: from equality (shell 0), q-weight j -> shell j; from complement (shell 6), q-weight j -> shell 6-j | Geodesics of the discrete chirality sphere |
| 43 | **GF(64) full finite field structure**: irreducible polynomial x^6 + x + 1, primitive element, Frobenius order 6, trace 32/32, subfield lattice | Verified |
| 44 | **GF(4) mode layer**: pair-level Frobenius coincides with global complement on Omega | Structural identification |

### A5. Permutation, Operator, and Shadow Structure

| # | Feature | Method |
|---|---------|--------|
| 45 | **128 distinct permutations** on Omega from 256 bytes, uniform 2-to-1 multiplicity | Exhaustive |
| 46 | **2 permutations of order 2** (S-gate), **126 of order 4** (all other bytes) | Exhaustive cycle typing |
| 47 | **Row-class theorem**: uniform transition matrix has exactly 32 distinct rows, rank 32; family-0 restriction gives 64 rows, rank 64 | Matrix computation |
| 48 | **8192-element operator family**: 4096 even-parity + 4096 odd-parity, semidirect product structure | Operator family tests |
| 49 | **Even operators as translations**: (tau_A, tau_B) covers the full C64 x C64 product | Verified |
| 50 | **Every word action is affine** on GF(2)^24 with identity or swap linear part | 500 random words |
| 51 | **Word signature composition**: sig(w1 o w2) = compose(sig(w2), sig(w1)) | 500 random word pairs |
| 52 | **16-to-1 multiplicity** from 65536 length-2 words to 4096 even signatures | Exhaustive |
| 53 | **Operator group**: G = (GF(2)^6 x GF(2)^6) rtimes C2; \|G\| = 8192; G' = Z(G) = diagonal GF(2)^6 (64 elements); abelian shadow G/G' = 128 | Algebraic + tests |
| 54 | **128 distinct next states** from any fixed state, uniform 2-to-1 multiplicity | All 256 bytes from fixed states |
| 55 | **Shadow partners**: b and b xor 0xFE produce the same Omega-permutation | Verified for substitution detection |
| 56 | **Global complement Z2 automorphism** commutes with all byte actions | Algebraic from XOR commutativity |
| 57 | **Spinorial double cover**: 256 SU(2) elements project to 128 SO(3) rotations | Structural theorem |
| 58 | **Dense operator generation**: 3729+ distinct signatures from 10,000 random length-3 words | Random sampling |

### A6. Depth-4 Closure and Commutativity

| # | Feature | Method |
|---|---------|--------|
| 59 | **b^4 = id** for every byte b on every state (order 4 universal) | All 256 bytes, exhaustive |
| 60 | **XYXY = id** for every byte pair (alternation identity) | All 65,536 ordered pairs |
| 61 | **T_b^2 is symmetric translation**: A and B shifted by the same amount | All bytes |
| 62 | **4-family cycle = global sign flip**: A4 = A0 xor 0xFFF, B4 = B0 xor 0xFFF | All micro_refs |
| 63 | **8-step closure**: applying the 4-family word twice returns to identity | 720-degree spinorial closure |
| 64 | **Depth-4 closed form** separates mask and family-phase contributions | 2000 random 4-byte sequences, zero failures |
| 65 | **Net family-phase invariants**: only (phi_a, phi_b) in (Z/2)^2 survives from 256 family combinations | All 4^4 combinations |
| 66 | **Depth-4 alternation explained by affine algebra**: swap^4 = id, translations cancel | Algebraic proof + 500 random pairs |
| 67 | **Discrete BCH theorem**: XYXY = id is the discrete realization of BCH depth-4 commutator cancellation from sl(2) | Exhaustive over 65,536 pairs |
| 68 | **1/64 commutativity rate** = 2^-6 (1024 commuting pairs out of 65536) | Exhaustive over all 256^2 pairs |
| 69 | **Every byte commutes with exactly 4 others** | Exhaustive |
| 70 | **Exact commutation condition**: bytes x, y commute iff q(x) = q(y) | 5000 random pairs |
| 71 | **Q-map**: 4-to-1 from the 256-byte alphabet onto C64 | Exhaustive |
| 72 | **Exact commutator defect formula**: K(x,y) translates by d = q(x) xor q(y), always in C64 | 5000 random pairs + exhaustive at rest |
| 73 | **Defect set = entire C64** | Exhaustive |
| 74 | **Q-fiber exact structure**: 256 bytes -> 128 Omega-maps -> 64 q-classes (4:1 then 2:1) | All 64 q-classes |
| 75 | **Each q-fiber has exactly 2 distinct Omega-signatures** | All 64 q-classes |
| 76 | **Fixed-x commutator defect multiplicity 4** | Exhaustive |

### A7. Future-Cone Entropy and Uniformization

| # | Feature | Method |
|---|---------|--------|
| 77 | **H0(s) = 0** for any s in Omega | Theorem + runtime check |
| 78 | **H1(s) = 7 exactly** for any s in Omega (128 distinct next states, uniform multiplicity 2) | Exhaustive |
| 79 | **Hn(s) = 12 exactly** for any s in Omega and n >= 2 | Exhaustive at n = 2; implied for n > 2 |
| 80 | **Exact 2-step uniformization**: every Omega state reached exactly 16 times from 65536 length-2 words | Exhaustive integer equality |
| 81 | **Exact per-byte capacity**: Shannon = min-entropy = 7.0 bits, zero variance | 500 sampled states |
| 82 | **Exact integer entropies**: H(state) = 12, H(state, parity) = 13, H(parity\|state) = 1, H(state\|parity) = 7 | Exhaustive over 256^2 words |
| 83 | **Parity adds exactly 1 bit** beyond the final state, uniformly across all states | 8192 distinct (state, parity) pairs |
| 84 | **Chirality and parity nearly independent**: mutual information ~ 0.014 bits | 200,000 random trajectories |
| 85 | **Witness synthesis**: every Omega state reachable in <=2 steps (1 at depth 0, 127 at depth 1, 3968 at depth 2) | Exhaustive; replay verified |

### A8. Quantum Information Protocols

| # | Feature | Method |
|---|---------|--------|
| 86 | **Graph state factorizes into 6 independent Bell pairs** (tensor product, exact to 10^-12) | 4 t-values, all 15 cross-pair marginals |
| 87 | **CHSH at Tsirelson bound 2*sqrt(2)**: primary quantum certificate on the Hilbert lift (see Formal Quantum Certification) | Precision 10^-12; correlators incompatible with local hidden variables |
| 88 | **No measurements exceed Tsirelson**: exhaustive angle grid (10^4 combinations) | Confirms 2*sqrt(2) as the hard quantum ceiling |
| 89 | **Exact quantum teleportation**: unique Pauli correction for all 8 (resource, outcome) combinations | 6 basis states + 800 random Bloch states; precision 10^-10 |
| 90 | **Monogamy**: same-pair pure, cross-pair maximally mixed, all 12 single-qubit marginals maximally mixed | Precision 10^-12 |
| 91 | **No-signalling**: Alice's marginal is independent of Bob's measurement choice | I2/2 in both Z and X bases, precision 10^-12 |
| 92 | **12 independent stabilizer generators**: all commute, GF(2) rank 12 | Precision 10^-12 |
| 93 | **64 X-translation elements** match C64; all stabilize the graph state | 256 random combinations |
| 94 | **Peres-Mermin contextuality**: row products +I, column 2 product -I | Precision 10^-12 |
| 95 | **Hilbert-lift entanglement**: XOR-graph subsets yield maximal reduced entropy (6 bits); Cartesian subsets yield near-zero | Bipartite von Neumann entropy |

### A9. Computational Quantum Advantages

| # | Feature | Method |
|---|---------|--------|
| 96 | **Hidden subgroup resolution in 1 step** (vs O(64) classical): q-map 4-to-1, WHT resolves the subgroup | Native q-map + WHT |
| 97 | **Deutsch-Jozsa in 1 step** (vs 33 classical): perfect discrimination, Pr = 1 for constant and balanced | All balanced functions tested |
| 98 | **Bernstein-Vazirani in 1 step** (vs 6 classical): all 6-bit secrets recovered with probability 1 | Multiple secret values |
| 99 | **Exact 2-step uniformization** (vs O(12) classical): exact uniform over 4096 states | Exhaustive verification |
| 100 | **Holographic compression**: 8 bits vs 12 bits per state (33.3% reduction) | Holographic dictionary |
| 101 | **O(1) commutativity decision** (vs 4 classical): compare q6(x) and q6(y) | 5000/5000 correct |
| 102 | **Universal period-4 holonomic closure** (depth-4 loop structure) | All bytes |
| 103 | **State separation**: every byte distinguishes every distinct state pair | 1000 sampled pairs x 256 bytes |
| 104 | **Hamming distance preserved** under every byte operation | 500 random triples |
| 105 | **Exact pairwise distance distribution**: C(12,k)/4096 at distance 2k; mean 12.0 | Exact from product structure |

### A10. Non-Clifford Resource and Universality

| # | Feature | Method |
|---|---------|--------|
| 106 | **BU dual-pole loop angle** delta_BU = 4·arctan(k(π/4)·k(m_a)) ≈ 0.195342178258 rad: representation-independent constant from depth-4 closure | CGM derivation + verification |
| 107 | **delta(BU) far from all Clifford angles**: nearest distance 0.195342 rad (multiples of pi/4) | All 8 Clifford angles tested |
| 108 | **No periodicity up to order 100,000**: closest return at k = 22,805, distance 7.62e-6 | Exhaustive search |
| 109 | **Dense U(1) equidistribution**: {k x delta(BU) mod 2pi} fills [0,2pi) uniformly; chi^2 = 0.232 vs critical 142.4 | 50,000 points, 100 bins |
| 110 | **Magic state Wigner negativity**: \|delta> has W(0,1) = -0.043771 | Discrete Wigner function computation |
| 111 | **Aperture gap** Delta = 1 - delta(BU)/m_a ≈ 0.020699545503: \|delta(BU) - m_a\| = Delta x m_a = 0.004128961943 exactly | Exact equality verified |
| 112 | **Three universality ingredients**: Clifford backbone, non-Clifford delta(BU), entangling gate S | Operator algebra + kernel tests |
| 113 | **Topological entanglement via holonomic gates**: localized A perturbation transported to B by gate S | Explicit mask 0x003 perturbation test |
| 114 | **Non-Clifford certification by 4 independent tests**: distance from Clifford, aperiodic spectrum, dense equidistribution, Wigner negativity | Each independently verified |

### A11. Error Detection, Tamper Provenance, and Non-Cloning

| # | Feature | Method |
|---|---------|--------|
| 115 | **Exact tamper detection (substitution)**: detected unless the replacement is a shadow partner; miss rate 1/255 | 50,000 trials |
| 116 | **Exact tamper detection (adjacent swap)**: detected unless q(x) = q(y); miss rate ~3/255 | 49,773 distinct pairs |
| 117 | **Exact tamper detection (deletion)**: detected unless the deleted byte is a gate stabilizer of the prefix state | 50,000 trials; misses only on horizons |
| 118 | **Exact perturbation rule**: payload bit flip = 1 chirality bit; boundary bit flip = 6 chirality bits; mean 2.25 | All 256 bytes, all 8 bit positions |
| 119 | **Ratio state_distance / chirality_distance = 2.000** constant over lengths 1 to 32 | Length-independent spreading |
| 120 | **Adversarial steering**: 16 byte-paths and 4 state-paths per target, exactly uniform | Exhaustive |
| 121 | **Horizon maintenance**: from the complement horizon, exactly 4/256 bytes keep the state on the horizon | All 64 horizon states |
| 122 | **Non-cloning**: transcription is fixed-point free; archetype 0xAA is the unique zero-intron source | All 256 bytes |
| 123 | **Equality horizon redundancy**: A = B adds zero information | Both components carry identical information |
| 124 | **Complement horizon relationality**: knowing A determines B uniquely | A = B xor 0xFFF |
| 125 | **Horizons structurally isolated** under all gate operations | All 4 gates verified |

### A12. Clifford Operator Algebra

| # | Feature | Method |
|---|---------|--------|
| 126 | **Byte actions are exact Clifford unitaries** over the self-dual code | Numerical verification |
| 127 | **Self-dual [12,6,2] code defines the stabilizer structure** for the graph state lift | Code + stabilizer tests |
| 128 | **Finite Weyl algebra** over GF(2)^6 with correct commutation relations | Algebraic verification |
| 129 | **Central spinorial involution** (frame operator quotient) | Operator family tests |
| 130 | **Depth-4 frame records strictly stronger than the final state** for genealogy | 100,000 random 4-byte words |

### A13. State Representation and Transcription

| # | Feature | Method |
|---|---------|--------|
| 131 | **24-bit GENE_Mac packing** (A12 << 12 \| B12) with exact round-trip | Pack/unpack tests |
| 132 | **Rest state 0xAAA555** with A xor B = 0xFFF at rest | Rest consistency |
| 133 | **Transcription involution**: byte_to_intron(byte_to_intron(b)) = b for all 256 bytes | All bytes |
| 134 | **256 distinct introns** (bijective transcription) | Enumeration |
| 135 | **Family from L0 boundary bits** (positions 0 and 7) | Bit-flip tests |
| 136 | **4 families x 64 micro_refs = 256** partition | Enumeration |
| 137 | **Palindromic intron structure** CS-UNA-ONA-BU-BU-ONA-UNA-CS | Structural |
| 138 | **Family acts only through the complement phase** during gyration | 4-family probe |
| 139 | **Dipole-pair mask expansion**: payload bit i toggles mask pair i only | All 64 micro_refs x 6 bits |
| 140 | **Reference byte 0xAA is pure swap** with 64 fixed points on Omega | Cycle census |
| 141 | **FIFO gyration spinorial cycle** (0, pi, 2pi, 3pi) from family bits | 4-phase verification |

### A14. Self-Dual Code and Mask Structure

| # | Feature | Method |
|---|---------|--------|
| 142 | **Self-dual [12,6,2] binary linear code** C = C perp | Set equality |
| 143 | **Pair-diagonal code**: every mask has pair-equal bits (00 or 11 per pair) | All 64 masks |
| 144 | **Weight enumerator** (1+z^2)^6: weights 0,2,4,6,8,10,12 with binomial counts | Exact enumeration |
| 145 | **Walsh spectrum** restricted to {0, 64}; support = C perp = C | All 2^12 positions |
| 146 | **Single-bit error detection**: all weight-1 errors detected (non-zero syndrome) | All 12 bit positions |
| 147 | **Undetected error enumerator** (1+z^2)^12: minimum undetected error weight 2 | Theoretical + sampled (512 states) |
| 148 | **Pair-flip errors stay in Omega** and produce C64 codeword displacements | Confirmed |
| 149 | **Erasure taxonomy**: 6 observed bit positions needed for unique codeword recovery | Exhaustive size-4 erasure census |
| 150 | **Pair erasure reduces rank by exactly 1** per erased dipole pair | Exhaustive |

### A15. Physical Constants

| # | Feature | Method |
|---|---------|--------|
| 151 | **Fundamental aperture constraint**: Q_G x m_a^2 = 1/2 | Exact algebraic identity |
| 152 | **Fine-structure constant prediction**: alpha_0 = delta_BU^4/m_a = 0.007299683573 (about +319.43 ppm vs CODATA); transport-corrected alpha = 0.007297352815 (about 33.7 ppb vs CODATA 2018); correction chain constant R = 0.993434896272 | Comparison with CODATA |
| 153 | **K_QG identity**: two derivations agree to <10^-12 | Numerical verification |
| 154 | **Stage action ratios**: E_ONA/E_CS = 1/2 exact; E_UNA/E_CS = 2/(pi*sqrt(2)) to 12 decimal places | Geometric values |
| 155 | **Aperture quantization chain**: 5/256 (byte) < Delta ≈ 0.020699545503 (continuous) < 1/48 (depth-4); 48·Δ ≈ 0.993578 | Three scales verified |
| 156 | **DOF doubling theorem**: 2^(2x1) = 4 (CS), 2^(2x3) = 64 (UNA), 2^(2x6) = 4096 (ONA) | BFS with restricted byte subsets |
| 157 | **Optical conjugacy on Omega**: constant density 0.5 at every state | Product structure U x V |

### A16. Hardware and Native Implementation

| # | Feature | Method |
|---|---------|--------|
| 158 | **C engine signature scan** matches the Python reference | Byte sequences |
| 159 | **WHT (wht64)**: orthonormal and self-inverse (max err ~2.38e-7) | vs reference matrix |
| 160 | **GyroMatMul GEMV**: vs torch.mv max abs err ~1.09e-5 | Numerical comparison |
| 161 | **Packed GEMV**: vs torch.mv max err ~7.45e-6; packed vs unpacked ~2.24e-6 | Numerical |
| 162 | **Operator projection basis**: project-reconstruct exact (max err ~5.96e-8) | Weyl/Heisenberg-Walsh basis |
| 163 | **OpenCL GPU vs CPU**: max err ~1.9e-6 | Cross-platform |
| 164 | **Target equivalence invariant**: all targets produce identical Results for the same circuit and initial state | Conformance requirement |
| 165 | **Two execution classes**: kernel-exact over GF(2)^24; tensor/spectral match the reference to specified tolerances | Two-class verification |

---

## Tier B: Mathematical Physics and Applications

The 247 features of this tier come from analyses in the science repository, each with a manuscript and a set of executable experiment scripts listed in the program index below. The analyses carry the kernel's verified structure into mathematical physics and other domains, from gravity and field theory to genomics.

### Program index

| Section | Analysis | Manuscript | Scripts | Features |
|---------|----------|------------|---------|---------:|
| B1 | Wavefunction | [Analysis_hQVM_Wavefunction.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_Wavefunction.md) | `hqvm_wavefunction_1.py`, `_2.py`, `hqvm_wavefunction_kernel.py` | 16 |
| B2 | Holonomy | [Analysis_Holonomy.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Holonomy.md) | `cgm_holonomy_analysis_1-2.py`, `_common.py`, `_run.py`, `hqvm_wavefunction_kernel.py` | 18 |
| B3 | Precession | [Analysis_Precession.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Precession.md) | `cgm_precession_analysis_1-2.py`, `_run.py` | 22 |
| B4 | Group theory | [Analysis_hQVM_CGM_Group_Theory.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Group_Theory.md) | `hqvm_group_analysis_1-5.py`, `_common.py`, `_run.py` | 25 |
| B5 | Gravity | [Analysis_Gravity.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Gravity.md) | `hqvm_gravity_analysis_1-10.py`, `hqvm_gravity_common.py`, `hqvm_gravity_runner.py`, `hqvm_corrections_analysis_1.py` | 62 |
| B6 | Electroweak masses | [Analysis_Compact_Geometry.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Compact_Geometry.md) | `hqvm_compact_geom_1-2.py`, `_common.py`, `_run.py` | 14 |
| B7 | Yang-Mills mass gap | [Analysis_hQVM_CGM_YM_Mass_Gap.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_YM_Mass_Gap.md), [Yang_Mills_Mass_Gap_Solution.md](https://github.com/gyrogovernance/science/blob/main/experiments/hQVM_CGM_YM_Gap/Yang_Mills_Mass_Gap_Solution.md) | `Yang_Mills_Mass_Gap_1-5.py`, `_common.py`, `_run.py` | 6 |
| B8 | Nuclear physics | [Analysis_hQVM_CGM_Trestleboard.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Trestleboard.md) | `hqvm_cgm_trestleboard_1-5.py`, `_run.py` | 8 |
| B9 | Percolation theory | [Analysis_hQVM_Percolation.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_Percolation.md) | `hqvm_percolation_analysis_1-5.py`, `_run.py` | 7 |
| B10 | Cohomology | [Analysis_hQVM_Cohomology.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_Cohomology.md) | `hqvm_Cohomology_analysis_1-4.py`, `_run.py` | 7 |
| B11 | Allometry | [Analysis_hQVM_CGM_Allometry.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Allometry.md) | `hqvm_cgm_allometry_1-3.py`, `_run.py` | 6 |
| B12 | Genomics | [Analysis_hQVM_CGM_Genomics.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_CGM_Genomics.md) | `hqvm_cgm_genomics_1-8.py`, `_common.py`, `_run.py` | 44 |
| B13 | Receipt geometry | [Analysis_hQVM_Moments_Fiat.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_hQVM_Moments_Fiat.md) | `hqvm_moments_fiat_analysis_1-3.py`, `_run.py` | 6 |
| B14 | Modal logic and 3D necessity | [Analysis_3D_6DOF_Proof.md](https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_3D_6DOF_Proof.md) | `cgm_3D_6DoF_analysis.py`, `cgm_3D_6DoF_helpers.py`, `cgm_axiomatization_analysis.py`, `cgm_modal_geometric_derivation.py` | 6 |
| **Total** | | | | **247** |

### B1. Wavefunction Analysis

| # | Feature | Method |
|---|---------|--------|
| 166 | **T1: K4 operator algebra {id, W2, W2', F}** for all 64 micro_refs on all 4096 states | 64 x 4096 exhaustive |
| 167 | **T2: W2 maps shell s to 6 - s** (pole swap, chi xor 63) | Algebraic proof + verified |
| 168 | **T3: W2' maps shell s to 6 - s** identically | Algebraic proof + verified |
| 169 | **T4: gate F preserves shell** (Z2 within pole) | chi xor 63 xor 63 = chi; verified |
| 170 | **T5: depth-4 confines to the opposite constitutional pole** | 64 x 64 states |
| 171 | **T6: F = W2 composed with W2'** is the K4 operator product of two depth-four half-words | Signature algebra |
| 172 | **T7: CS forces canonical family ordering** | 64 micro_refs |
| 173 | **T8: BU-Egress = W2 involution** (depth-4 squares to identity on Omega) | 4096 states + complement horizon |
| 174 | **T9: BU-Ingress = W2 pole-pairing** (shadow = memory) | 4096 states |
| 175 | **T10: q(W2) = q(W2') = 63; q(F) = 0** for all m | Algebraic proof from L0 parity |
| 176 | **Eigenspace decomposition** under U_W: dim(+1) = 2048, dim(-1) = 2048 | Spectral computation |
| 177 | **Gate F is a fixed-point-free involution** on all 4096 states (2048 two-cycles, 0 fixed points) | Exhaustive |
| 178 | **Z2 oscillation**: rest to swapped with period 2 in word-count | Carrier trajectory |
| 179 | **Constitutional trajectory per 4-byte turn**: shells [0,1,6,1,0] symmetric about the equality transit | Byte-by-byte tracking |
| 180 | **Carrier Z2 coordinate** within each shell: rest vs swapped, invisible to chirality | Gate F as Z2 flip |
| 181 | **Egress and Ingress as dual readings of one W2 operator** | Structural theorem |

### B2. Holonomy Analysis

| # | Feature | Method |
|---|---------|--------|
| 182 | **Stage-angle identities**: theta_CS = theta_UNA + theta_ONA and theta_CS + theta_UNA + theta_ONA = pi, exact from thresholds (pi/2 + pi/4 + pi/4 = pi) | Exact threshold algebra |
| 183 | **Exact SU(2) commutator holonomy** phi_SU2 = 2 arccos((1 + 2 sqrt(2))/4) = 0.5879007626540203 rad (33.6842 deg); the 80-digit matrix computation matches the closed form with residual 7.4e-81 | Closed form vs matrix |
| 184 | **Gyration machinery calibrated against Thomas-Wigner**: slope error scales as beta^2 (measured order 2.003), residual as beta^4 (order 4.006); about 1e-8 agreement at stage coordinates | Scaling census |
| 185 | **delta_BU corner verified by four independent constructions** (raw gyration map, Ungar SO(3) closed form, Lorentz factorization spatial block, analytic Wigner) at the 1e-81 floor | Four-route identity |
| 186 | **Origin-gyr word** R = G_ingress G_middle G_egress: the middle factor is the identity (collinear poles), the corners share an axis, conjugacy angle = delta_BU | Word factorization |
| 187 | **Palindromic conjugation theorem** H_pal = A^-1 H_BU A with A = gyr(UNA, ONA): angle preserved at delta_BU, axis transported to (-0.9224, 0.3863, 0) | Conjugation identity at 80 digits |
| 188 | **Ungar inversion identity** gyr(v, u) = gyr(u, v)^-1 on the return leg | Inversion check |
| 189 | **Gyrotriangle defect identities**: defect(0, ONA, BU+) = omega = delta_BU/2; defect(ONA, BU+, BU-) = delta_BU, equal to hyperbolic area in curvature units; gyrogroup axioms hold on (UNA, ONA, BU+) | Defect census |
| 190 | **Mass-shell geodesic holonomy**: path products certified in SO+(1,3) (metric norm, determinant, orthochronous residuals at the floor); conjugacy angle equals delta_BU on the dual-pole loop and the palindrome | Lorentz path products |
| 191 | **Circular calibration** alpha_circ(V) = 2 pi (gamma(V) - 1) at the working floor for V in {0.1, 0.2, 0.3, 0.4, 0.6} | Circular Thomas check |
| 192 | **Cartesian Thomas path-ordered exponential** with Richardson extrapolation recovers delta_BU; spherical z-chart readout 0.2466 with offset G_z = 0.0512616448; the relative-boost word 0.2585 is a distinct path object (differs by about 0.063) | Transport prescriptions |
| 193 | **Aperture decomposition**: rho(0) = 2 k(pi/4) = 0.9702317252823254; baseline gap 0.0297682747176746; finite-amplitude correction 0.0090687292150043; final Delta 0.0206995455026703; rho even in m_a; baseline gap positive iff beta < 4/5 | Closure series |
| 194 | **Byte fold distribution** N(k) = 16 x C(4, k) = (16, 64, 96, 64, 16); 16 flat bytes, 240 curved; the central BU comparison disagrees in 128 of 256 bytes | Byte census |
| 195 | **W2 / W2' K4 certificate** on all 4096 states in exact arithmetic: involutions, shell map s to 6 - s, chirality xor 63; independent reproduction of B1 | Exhaustive K4 |
| 196 | **Byte-horizon aperture quantization**: 256 x Delta = 5.299083648683975, round = 5, Q_256(Delta) = 5/256 with relative error 0.05645; shared constant APERTURE_GAP_Q256 = 5 | Horizon quantization |
| 197 | **Natural scale triple**: 5/256 (byte horizon), 1/48 (depth-4, 48 x Delta = 0.9935781841281744), 1/32 (turn-normalized delta_BU / 2 pi); ratio (1/48)/(1/32) = 2/3 | Scale chain |
| 198 | **Continuous-finite correspondence**: 12 structural rows (closed path to operator word, holonomy angle to involution, BU loop to W2 exchange, Delta to 5/256, palindrome to byte fold, CS frame to boundary bits, conjugacy spectrum to +1/-1) | Correspondence table |
| 199 | **Sensitivity elasticities**: (theta_ONA / delta_BU) d(delta_BU)/d(theta_ONA) = 1.61296528 and (m_a / delta_BU) d(delta_BU)/dm_a = 1.01888667; finite-difference derivatives match | Elasticity audit |

### B3. Precession Analysis

| # | Feature | Method |
|---|---------|--------|
| 200 | **Three elementary pair precessions** omega_UO = 0.396601502221 (22.7236 deg), omega_OB = 0.097671089129 (5.5961 deg), omega_UB = 0.083413894169 (4.7793 deg); each equals the Ungar gyrotriangle defect of its origin triangle; three mutually orthogonal rotation axes | Pair defect census |
| 201 | **Half-angle product identities**: tan(omega_UO/2) = k(UNA) k(ONA) = 0.200941569628, tan(omega_OB/2) = 0.048874404435, tan(omega_UB/2) = 0.041731146576 | Product identities |
| 202 | **Inversion of the three identities** recovers the Poincare radii: k_UNA = sqrt(ac/b), k_ONA = sqrt(ab/c), k_BU = sqrt(bc/a) | Radius recovery |
| 203 | **Closed forms at UNA**: k(1/sqrt(2)) = sqrt(2) - 1 (silver-ratio conjugate); proper velocity gamma beta = 1, the Compton-momentum condition p = mc | Exact UNA forms |
| 204 | **Rotation-channel dual-pole holonomy** delta_UNA_BU = 2 omega_UB = 0.166827788338, orthogonal in axis to the ONA-rooted channel | Dual-pole channel |
| 205 | **Channel ratios**: delta_UNA_BU / delta_BU = 0.854028504372 vs tan(omega_UB/2)/tan(omega_OB/2) = k_UNA/k_ONA = 0.853844605530; distinct identities differing by about 2 parts in 10^4 | Ratio comparison |
| 206 | **Spinorial double-cover relation** delta_BU / 2 = omega_OB | Double-cover identity |
| 207 | **Palindrome steering**: R_pal = A^-1 R_BU A; axis transport angle = omega_UO at full strength; full-turn period 2 pi / omega_UO = 15.8426 conjugations | Steering theorem |
| 208 | **Closed-walk census** (all closed walks of length 2 to 5): exactly six Fermi-Walker holonomy values: 0 (72 walks), 0.166827788338 (66), 0.195342178258 (66), 0.256712834405 (8), 0.412719054050 (16), 0.420475081676 (132); each a word in {UO, OB, UB, id} | Walk enumeration |
| 209 | **Noncommutativity residual**: delta_UOB - (omega_UO + omega_OB + omega_UB) = -0.157211403843 | Residual angle |
| 210 | **Intrinsic transport equivalence** on all 360 enumerated walks: Fermi-Walker = origin gyration = geodesic transvection = gyrotriangle defect; the Cartesian Thomas path-ordered exponential equals the geodesic angle to 9 decimals on the BU loop and the palindrome; vanishes identically on out-and-back paths | Transport census |
| 211 | **Inertial-frame boost composition**: every bent walk leaves residual velocity, only the 4 collinear BU pole exchanges close to pure rotation; length-5 walks give 147 distinct rotation angles vs 6 intrinsic; residual identities F = theta_inert - theta_can and theta_RF (F_BU = 0.063136496377; on the BU loop theta_RF = theta_can + theta_inert) | Inertial residual |
| 212 | **Chart readouts**: z-chart 0.2466038230 (offset 0.0512616448), diagonal 1.24655632 (offset 1.05121414), difference 0.99995249 rad; x and y chart values tabulated; the Cartesian chart is regular and intrinsic | Chart comparison |
| 213 | **Closure response decomposition**: linear gain rho_0 = 2 k(o_p) = 0.970231725282 (Delta_0 = 0.029768274718); secant rho = 0.979300454497 (rho - rho_0 = 0.009068729215, Delta = 0.020699545503); tangent rho_tan = 0.997796177590 (Delta_tan = 0.002203822410); both budget identities close; Delta / Delta_tan = 9.393 | Gain budget |
| 214 | **Elasticities**: E_ONA = 1.612965279633, E_BU = 1.018886668547, ratio 1.583066428706; the translational threshold steers the holonomy superlinearly | Elasticity audit |
| 215 | **Equal-speed Wigner calibration**: omega_0 = TW(u_p, u_p; o_p) = 0.215549910153 with partial derivatives (12 sqrt(2) - 4)/17 = 0.762974279322 and (21 - 12 sqrt(2))/17 = 0.237025720678, summing to 1 exactly | Wigner calibration |
| 216 | **Aperture clock and stage actions**: t_aperture = m_a; Q_G = L_horizon / m_a = 4 pi; S(delta_BU) = rho; S_CS = 7.874805, S_UNA = 3.544908, S_ONA = 3.937402 = K_QG, S_BU = m_a, S_GUT = 1.508167; EM duality angle arctan(S_ONA/S_UNA) = 48.003 deg | Stage action table |
| 217 | **Delta-ruler coordinates** n = theta / Delta: omega_UB 4.030, omega_OB 4.719, delta_UNA_BU 8.059, delta_BU 9.437, omega_UO 19.160, phi_SU2 28.402 | Coordinate table |
| 218 | **Electron Compton conversion** of the balance channel: T_aperture = 2.569365e-22 s, f_BU = 1.210014e20 Hz, Omega_BU = 7.602741e20 rad/s, E_BU = 5.004215e5 eV, E_aperture = 1.057745e4 eV (from T_C = 1.288089e-21 s) | Compton conversion |
| 219 | **Equivalent circular Thomas speeds** by inverting 2 pi (gamma - 1): beta = 0.161344 (omega_UB), 0.174297 (omega_OB), 0.243712 (delta_BU), 0.339443 (omega_UO), 0.404725 (phi_SU2) | Circular speeds |
| 220 | **Compact fiber mismatch**: 3 delta_BU = 0.586026534773; epsilon_CH = phi_SU2 - 3 delta_BU = 0.001874227881; sigma = epsilon_CH / m_a = 0.009395985; axis inner product -0.357407 (chi_CH = 69.059 deg); rel(C, U_BU^3) = 0.962562, rel(C, UxUyUz delta) = 0.482226 | Fiber mismatch |
| 221 | **Realization controls**: rapidity placement gives delta_BU = 0.148518013495; equalized large speeds force omega_UB = omega_OB and raise the steering quantum to 0.462263357559; axis orthogonality persists in all placements; the Einstein-speed placement is the gravity and EM anchor | Placement controls |

### B4. Group Theory Analysis

| # | Feature | Method |
|---|---------|--------|
| 222 | **Wreath-product presentation**: G = GF(2)^6 wr C_2, equivalently six D8 mode copies sharing one global exchange bit, \|G\| = 2 x 4^6 = 8192 | Group order census |
| 223 | **Rest-state stabilizer** {id, z} with z the exchange plus all-ones translation (1, 63, 63); orbit-stabilizer 8192/2 = 4096; Omega = G/H homogeneous space | Stabilizer theorem |
| 224 | **Every byte action is an even permutation**: T_b in A_4096 | Parity census |
| 225 | **Byte-graph automorphisms**: chirality-compatible subgroup of order 92160 = 128 x 720; step-preserving subgroup 46080 = 64 x 720 | Automorphism census |
| 226 | **Palindrome symmetry group** B_3 = C_2 wr S_3 of order 48; the orientation-preserving intersection is S_4 (order 24); the binary octahedral 2O occupies the spin double-cover slot | Symmetry identification |
| 227 | **Conjugacy-class census**: 2144 classes = 64 central (size 1) + 2016 paired translation (size 2) + 64 exchange (size 64); 2016 = C(64, 2) | Class equation |
| 228 | **Irreducible representation census**: 128 linear + 2016 two-dimensional; 128 x 1^2 + 2016 x 2^2 = 8192; the two-dimensional sectors arise by Clifford induction | Irrep census |
| 229 | **Bidegree classification** of two-dimensional irreps by character-weight pairs with closed-form counts C(6, n_u) C(6, n_v) and [C(6, n)^2 - C(6, n)]/2 | Bidegree formulas |
| 230 | **Multiplicity-free carrier decomposition**: L^2(Omega) = 64 linear + 2016 two-dimensional sectors, 64 + 2016 x 2 = 4096; the 64 H-odd linear characters are absent from the projected manifold | Carrier decomposition |
| 231 | **(G, H) is a finite Gelfand pair**: 2080 double cosets, one H-fixed vector per appearing irrep (zonal spherical functions) | Gelfand pair check |
| 232 | **Commutant dimension**: dim End_G(L^2(Omega)) = 2080; equivariant compression hierarchy 16,777,216 dense / 4096 translation / 2080 full G / 7 shell-radial | Endomorphism hierarchy |
| 233 | **32-bit register lift**: register32 = intron8 or state24; 64 H-odd functions psi_a mutually orthogonal with inner product 8192 delta_ab; shadow-sheet phase recovery | Lift orthogonality |
| 234 | **Finite Fourier analysis on G**: matrix transform, Plancherel identity, and inversion over all 2144 irreps | Plancherel audit |
| 235 | **Walsh characters** W_k(x) = (-1)^(k·x) give the exact Fourier basis of the 4096-element translation subgroup | Abelian Fourier |
| 236 | **Two-step mixing as an operator identity**: P^2 = J/4096; rank(P) = 32, rank(P^2) = 1; transient image indexed by GF(2)^6 / <111111> | Mixing identity |
| 237 | **Nilpotent transient structure**: N^2 = 0 with rank 31 and 31 length-two Jordan chains; exact mixing by algebraic annihilation, contrasted with the SO(3) heat kernel spectral gap lambda_1 = 2 | Jordan structure |
| 238 | **Sharply uniform two-byte design**: 16 ordered witnesses per source-target pair; algebraic router with provenance diversity and load distribution | Design census |
| 239 | **Word compilation**: any byte word compiles to a 13-bit signature (8192 = 2^13); interpreted cost proportional to nB vs compiled cost proportional to n + B | Compilation cost |
| 240 | **Shell thermodynamics from group factorization**: Z_1(lambda) = 64 (1 + lambda)^6; E[N] = 6 lambda/(1 + lambda); Var(N) = 6 lambda/(1 + lambda)^2; horizon variance zero, equator maximal | Partition function |
| 241 | **Carrier trace theorems**: Tr(M_2k) = 7/(2k + 1) by Chu-Vandermonde; odd-weight traces zero; return traces via Krawtchouk eigenvalues, verified by three independent computational routes | Trace identities |
| 242 | **Grover search geometry on the lift**: F^2 = I with eigenspaces 2048/2048; success probability sin^2((2k+1) theta) with sin^2 theta = M/N | Grover geometry |
| 243 | **Fourier synchronization**: the abelian Walsh sector and the nonabelian matrix-Fourier sector recover alignment with distinct noise tolerance | Sync comparison |
| 244 | **SO(3) rotation codebooks**: encoder Omega to SO(3) and quantizing decoder; 24-bit budget mean geodesic error 0.0114 rad; rigid-body composition chains within budget | Rotation codebook |
| 245 | **Central character phase readout** chi_a(d) = (-1)^(a·d): scalar phases read the six-dimensional native holonomy | Phase readout |
| 246 | **Exterior grading**: dim Lambda^k(GF(2)^6) = C(6, k), vanishing alternating sum (Euler characteristic), discrete Poincare duality | Exterior algebra |

### B5. Gravity Analysis

The gravity analysis links the kernel to gravitation in two layers: discrete invariants of the kernel that anchor the theory, and a continuous field theory with its predictions.

#### Kernel invariants

| # | Feature | Method |
|---|---------|--------|
| 247 | **Shell displacement invariant D = 24** across all 64 mass configurations | Kernel census |
| 248 | **Discrete Gauss law**: G_kernel = Q_G / D = pi/6 | Q_G x G_kernel = D |
| 249 | **Plaquette curvature spectrum**: 1024 x C(6,k) for popcount k = 0 to 6 | Exhaustive over 256^2 byte pairs |
| 250 | **Codeword-pair curvature census**: 64 x C(6,k) over the 64^2 mask codeword pairs, the same binomial as the shell spectrum | Code-level curvature identity |
| 251 | **Plaquette census reproduces D = 24**: sum of popcounts / (2\|Omega\|) = 24 | Closed-form calculation |
| 252 | **Closed-form popcount sum**: total defect weight 196608 = 1024 x 6 x 2^5, giving D = 196608 / (2 x 4096) = 24 exactly | Independent Gauss-law route |
| 253 | **Refractive Depth as Regge action**: tau_G matches the closed form \|Omega\| Delta rho^5 (1 - 4 rho Delta^2) to relative precision 3.7e-16 | Executable verification |
| 254 | **Per-cycle Regge values**: S_cycle = 0.100300491235, tau_cycle = 0.021256806515, N_cycles = 3586.52; the product equals tau_G exactly | Cycle decomposition |
| 255 | **k_eff = 3 from the Regge sum**: spatial dimension emerges from BCH closure | Numerical readout |
| 256 | **Z2 BCH selection rule**: only even-order corrections survive projection | Symbolic computation (Dynkin truncation) |
| 257 | **Antimatter gravitoelectric invariants even**: D = 24 holds for matter and antimatter | Exhaustive over 4096 states |
| 258 | **Antimatter gravitomagnetic invariants odd**: H_spin(C(s)) = -H_spin(s) for 2816 non-equatorial states | Exhaustive computational verification |
| 259 | **Constant-product identity**: alpha_0 zeta = rho^4 / (pi sqrt(3)) independent of m_a | Algebraic cancellation |
| 260 | **Isotropic trace scalar**: tau_trace = \|Omega\| Delta rho^5 c_4 Delta^4 with c_4 = -7/4 fixed by two routes; monopole sector separate from the STF attenuation | Dual-route identity |

#### Field theory and predictions

| # | Feature | Method |
|---|---------|--------|
| 261 | **Q_G = 4 pi as quantum of gravity** (horizon normalization) | GNS + kernel ratio |
| 262 | **Virial condition 2T + V = 0** as structural consequence of ancestry preservation | Kernel invariant D = 24 |
| 263 | **Transport-corrected fine-structure constant** alpha = 0.007297352815, about 33.7 ppb from CODATA 2018, via three geometric corrections in powers of Delta (Thomas-Wigner curvature ratio R = 0.993434896272) | Correction chain vs CODATA |
| 264 | **Delta self-consistency**: 3-factor reconstruction converges; D^3 fixed-point residual < 10^-15 | Iterative computation |
| 265 | **Position-dependent coupling**: G(psi) = G0 exp(g1 psi) with g1 = -0.6456 | Three independent routes |
| 266 | **Weak-field G residual about +2.99 ppm vs CODATA** (tau_G - tau_required = -2.99e-6; CODATA uncertainty about 22 ppm) | G_pred = G_kernel exp(-tau_G)/v^2 |
| 267 | **c4 = -7/4** fixed by two independent kernel routes | STF + closure charge |
| 268 | **Per-family Refractive Depth uniformity**: zero variance across all 4 families | Verified |
| 269 | **Exact point-mass solution**: psi(s) = -(1/g1) ln(1 - g1/s) | Analytical + numerical endpoints |
| 270 | **Effective metric**: f = 1 - 2 psi; Einstein tensor verified to 4.4e-16 | Numerical |
| 271 | **Modified Gauss law conservation** at all radii to 2.83e-16 | Numerical |
| 272 | **Self-energy theorem**: E_self = -M c^2 / 4 (exact, finite) | Exterior ODE |
| 273 | **Mass dressing**: M_obs = (4/5) M_bare (20 percent bound into field) | Self-consistent |
| 274 | **Chiral correction magnitude**: (4/75) psi^2 from the constant anisotropy ratio | Kernel invariant 2/75 |
| 275 | **PPN: gamma = 1** exactly (consistent with Cassini) | Leading deflection |
| 276 | **Nordtvedt parameter eta_N = 0** | G(psi) position-only dependent |
| 277 | **Mercury precession**: CGM/GR = 0.9999999973 (0.003 ppm) | Full metric geodesic |
| 278 | **Black hole shadow**: CGM predicts 80 percent of the GR Schwarzschild area | Null geodesic computation |
| 279 | **Horizon at s_h = 1.695 r_g** (15.3 percent inward of Schwarzschild) | psi = 1/2 condition |
| 280 | **Photon sphere at s_ph = 2.586 r_g** (vs 3.0 in GR) | Null geodesic |
| 281 | **Gravitational radiation**: quadrupole dominant; exactly 2 tensor polarization modes | Fourier decomposition |
| 282 | **Gravitational wave phase correction**: about -6.5 percent at v/c = 0.4 (GW150914) | Leading post-Newtonian |
| 283 | **Ringdown frequency shift**: fundamental about 12.5 percent above GR | Regge-Wheeler potential |
| 284 | **Vacuum impedance matching**: R + T = 1 across sharp metric steps | Numerical integration |
| 285 | **UV-IR interface density depletes by ~10^-6 near the horizon** | From E_ref formula |
| 286 | **Inflationary observables**: n_s = 0.972, r = 2.4e-3 in the R^2 limit | Slow-roll computation |
| 287 | **Asymptotic freedom of gravity**: d ln alpha_G / d ln mu = -0.017 < 0 | Refractive Depth law |
| 288 | **Neutron star TOV with G(psi)**: R = 15.4 km, M = 1.25 M_sun for gamma = 2 polytrope | Numerical integration |
| 289 | **Redshift prediction for NS surface**: z_CGM = 0.200 vs z_GR = 0.235 | Direct from metric |
| 290 | **Four-phase causal cycle**: Measure (CS), Vary (UNA), Retrieve (ONA), Commit (BU) | Byte transition decomposition |
| 291 | **E^2/5 efficiency**: rest-frame energy = M_obs c^2/4 = (1/5) M_bare c^2 | From self-energy theorem |
| 292 | **Intrinsic gravitational clock**: T_Z2 = (6/pi) G M/c^3 x surface gravity; vanishes at psi = 1/2 | D = 24 tied to speed of light |
| 293 | **Stage mass decomposition**: f_UNA = 0.462, f_ONA = 0.513, f_closure = 0.026; gravitating sector fraction 0.974; dressing uniform across components | Component mass fractions |
| 294 | **Gravitational-wave memory as interrupted holonomy**: a wave that interrupts the closed two-pass cycle leaves a residual rotation, the discrete precursor of memory | Holonomy interruption |
| 295 | **Radiation spectrum detail**: dominant mode \|A2\| = 1.25; hexadecapole precursor \|A4\| = 1.02 (82 percent of \|A2\|); two equal peaks per cycle | Fourier mode census |
| 296 | **Regge-Wheeler barrier peak** at s = 2.84 r_g vs 3.28 r_g in Schwarzschild | Potential peak location |
| 297 | **Horizon thermodynamics**: surface gravity = 1.01 kappa_GR; horizon area 72 percent; Hawking luminosity ratio 0.74 | Horizon readout |
| 298 | **EHT spin-sector shadow estimates**: M87* 36.2 μas and Sgr A* 48.0 μas at the measured spin priors | Null-geodesic computation |
| 299 | **Coupling reduction across objects**: Earth -0.45 ppb, Sun -1.4 ppm, white dwarf -0.019 percent, NS surface -9.4 percent (TOV) to -10.5 percent (Newtonian), stellar BH horizon -27.6 percent | G(psi) object survey |
| 300 | **Redshift deficit table**: for 1.4 M_sun, delta z / z_GR from -8.7 percent (15 km) to -22 percent (8 km); observable at psi > 0.1 | Compact-object redshift |
| 301 | **Propagation speed**: the gravitoelectromagnetic wave speed c is consistent with the GW170817 bound below 3e-15 | Speed bound |
| 302 | **Fine-structure cosmological modulation**: period Delta = 0.0207 in ln(1+z), fractional amplitude 4.8e-4, 7 sub-cycles (sub-period Delta/7 = 0.0030) | Shell-opacity prediction |
| 303 | **Laboratory G method dependence**: per-family Refractive Depth variance exactly zero; method-to-shell projection as the quantitative test for inter-method G scatter | Family uniformity |
| 304 | **Inflationary closure details**: A_s = A_s^pl Pi_H with Pi_H = rho^8 Delta^4 / (pi^2 \|Omega\|); 1/xi_eff = 8.40e-3; N_eff = 10^5 to 10^6 from the 32-bit lift quotient | Slow-roll closure |
| 305 | **Finite Weyl pair and uncertainty on GF(2)^6**: M_k T_a = (-1)^(k·a) T_a M_k and \|supp Psi\| x \|supp Psi_hat\| >= 64 for all nonzero state functions | Finite uncertainty theorem |
| 306 | **Vacuum refractive index** n = 1/sqrt(1 - 2 psi) with constant wave impedance across sharp metric steps; zero interface reflection, all vacuum reflection is smooth tunneling | Refractive vacuum |
| 307 | **Scalar-tensor classification**: effective scalar phi = exp(-g1 psi) with formal omega_BD = 0 under algebraic slaving; zero scalar radiation, eta_N = 0, gamma = 1 | Comparison with Brans-Dicke theory |
| 308 | **Nariai bound match**: interior anisotropy ratio sqrt(6)/9 = 0.2722 equals the Nariai ultracold mass bound for stable extremal compact objects | Anisotropy identity |

### B6. Electroweak Mass Analysis

| # | Feature | Method |
|---|---------|--------|
| 309 | **Carrier-trace polynomial** for top, Higgs, Z, W masses with 6 coefficient orders (Delta through Delta^5) | Fixed discrete grammar |
| 310 | **Max tick error 6.15e-9** at fifth order across four channels | Comparison with PDG |
| 311 | **W/Z ratio recovers Delta to 8.34e-10** | W/Z split back-solve |
| 312 | **Leave-one-out prediction**: each of H/Z/W predicted from the other two to ~10^-5 relative | Cross-validation |
| 313 | **Null-model audit**: rank-1 assignment gap ~11,000x over rank-2 | Exhaustive over 4096 flag assignments |
| 314 | **Coefficient admissibility**: structural audit with discrete grammar | Structural audit |
| 315 | **Trace-free conditions**: Sum p_i = 0, Sum q_i = 0 | Algebraic |
| 316 | **Coupling parametrizations**: lambda_H, g, g_Z, g', e, alpha_EW Delta, y_t to ~10^-5 relative | From mass law at tree level |
| 317 | **Lepton carrier layer**: tau, mu, e coordinates via M_shell; unique path (5, 8, 14) | Exhaustion over 680 valid triples |
| 318 | **148/51 closure**: K4 depth-4 (128) + full-byte len-2 (16) + micro paths (4) = 148 | Exact rational |
| 319 | **Archetype closure**: electron dyadic closes at -51/256 | Exact rational |
| 320 | **D_flow^2 quark ladder**: exact squared spacing \|d_flow\| = 1 to 6 for 6 quarks | Empirical |
| 321 | **UV-IR conjugacy**: E_UV x E_IR = E_CS x v/(4 pi^2) at all 4 stages | Product = K to 9+ digits |
| 322 | **SU(3) sextet bracket closes** in 32-bit lifted space | Phase-symmetrized check |

### B7. Yang-Mills Mass Gap Analysis

| # | Feature | Method |
|---|---------|--------|
| 323 | **Oriented aperture** Delta = 1 - delta_BU/m_a = 0.020699545503; discrete anchor 5/256; depth-4 alignment 48 Delta = 0.993578 | Direct carrier computation |
| 324 | **Unoriented shadow formula** Delta_W(n) = n/(2(n-1)) approaching 1/2, distinct from the oriented aperture regime | Shadow formula |
| 325 | **Carrier commutator defect**: commuting fraction 1/64; grade-2 multiplicity C_2 = 15 | Defect spectrum census |
| 326 | **Mass estimate** m_gap = C_2 v Delta^2 = 1.582473 GeV, within the lattice light-scalar glueball window | Saturated grade-2 multiplet |
| 327 | **Independent normalization cross-check**: 1.661555 GeV (relative deviation 4.76 percent) | Independent normalization |
| 328 | **Defining Q_8 Wilson chart, Aut(Q_8) symmetry, OS Gram positivity** on audited finite charts | Wilson/OS certificates |

### B8. Nuclear Physics

| # | Feature | Method |
|---|---------|--------|
| 329 | **W/Z mass-ratio aperture lock**: the aperture implied by the W/Z mass ratio agrees with the reference aperture to 8.34e-10 absolute error | Mass-ratio back-solve |
| 330 | **Th-229m optical isomer**: E_min = 8.3563 eV vs Zhang CaF2 8.3557335(8) eV (rel error 7.19e-5) | Forced class (6,2) |
| 331 | **Deuteron binding**: E_d = v Delta^3 + v Delta^4 (2/sqrt(5)) = 2.2242 MeV vs PDG 2.2240 MeV (rel error 8.89e-5) | Strong bare + tensor correction |
| 332 | **Alpha Gate F**: shell, shell-parity, daughter \|N - Z\| mod 7 preserved on 314/314 LiveChart alpha parents | Carrier-word census |
| 333 | **Beta routing**: shell-parity on 801/801 beta-minus parents; daughter J agreement 402/402 on the depth-1 stratum | IAEA LiveChart census |
| 334 | **Fusion map**: barriers for seven fuels on k = 3 strong-family; 5/7 literature resonances at percolation landmarks | S-factor holdout tables |
| 335 | **Magic numbers 2, 8, 20, 28, 50, 82, 126** from mixed Nilsson at (kappa, mu) = (1/32, 1/5) with left chirality | Gap-closure ranking |
| 336 | **Chirality flip removes intruders 28, 50, 82, 126** from the mixed large-gap dominant set | Ancestry-preservation bias test |

### B9. Percolation Theory Analysis

| # | Feature | Method |
|---|---------|--------|
| 337 | **Square-Root Cluster Theorem**: \|Reach_d(A)\| = (2^r(A))^2 under fiber-complete restriction | Verified d = 1 to 8 (52/52 gates) |
| 338 | **Byte regime**: unclosed spinorial half-cycles connect maximally on the full 4096-state product | Exhaustive reachability census |
| 339 | **Word regime**: depth-4 closure confines reachability to 128 horizon states from rest | Canonical word operators |
| 340 | **Five coverage observables** turn on at separable generator fractions on one restriction dial | Exact threshold labels |
| 341 | **Exact rank thresholds at d = 6**: micro-reference p_c = 0.0908, Q6-class p_c = 0.1053 | GF(2) rank machinery |
| 342 | **hQVM(d) family**: closed-form register-protocol thresholds and asymptotic square-root scaling | Finite-size scaling suite |
| 343 | **Gravity bridge**: percolation transport closes to gravitational self-energy identities | Structural observables |

### B10. Cohomology Analysis

| # | Feature | Method |
|---|---------|--------|
| 344 | **Shell census from exterior-algebra grading**: 64, 384, 960, 1280, 960, 384, 64 with discrete Poincare duality | Graded dimension derivation |
| 345 | **Parity 1-cocycle**: even-weight restriction confines reachability to even shells (32^2 = 1024 cluster) | Kernel excludes odd shells |
| 346 | **H^1(K4, GF(2)^6) family-fiber cohomology** classifies generator-restriction obstructions | Group cohomology census |
| 347 | **Grothendieck constant K_G^R(2) = sqrt(2)**: Boolean CHSH 2 vs Hilbert lift CHSH 2 sqrt(2) on the horizon ensemble | Walsh vs Hilbert comparison |
| 348 | **CHSH gap localizes to the 2 x 2 projection** (the full 63 x 63 observable matrix gives ratio 1) | Block projection audit |
| 349 | **Lefschetz census**: 252/256 bytes have zero fixed points; 4 bytes fix 64 states each | Fixed-point enumeration |
| 350 | **Aperture bridge**: Delta = 1 - delta_BU/m_a identifies the finite transport obstruction with the BU closure residual | Obstruction scalar link |

### B11. Allometry Analysis

| # | Feature | Method |
|---|---------|--------|
| 351 | **Channel basis at d = 6**: a_SR = 1/2, a_surf = 2/3, a_bulk = 3/4, a_time = 1/4, a_service = 1/12 | Source-accessibility exponents |
| 352 | **QuBEC thermal point** <N> = d/2 = 3; a_bulk = <N>/(<N>+1) = 3/4 (Kleiber) | Shell moment M_shell = 192 |
| 353 | **Three hQVM(d) consistency relations** (Rel I to III) each lock uniquely at d = 6 | Family consistency gates |
| 354 | **Chemical clock at T = 310 K**: E_a = 0.645 eV inside the 0.6 to 0.7 eV MTE band | Delta-ruler activation energy |
| 355 | **Catalog Kleiber audits in the mu-band [2/3, 3/4]**: PanTHERIA BMR OLS 0.717; AnAge metabolic OLS 0.713 | OLS/RMA with bootstrap |
| 356 | **Damuth dual-null pattern**: population density OLS -0.741 near -3/4; RMA -0.980 near -1 | External trait catalogs |

### B12. Genomics Analysis

| # | Feature | Method |
|---|---------|--------|
| 357 | **Encoding orbits**: 24 affine nucleotide encodings in three orbits of 8 charts; Watson-Crick, transition, and transversion act by translation on every chart | Chart orbit census |
| 358 | **Pair-inversion orbit**: WC polarity is the six-bit antipode 111111; fold and payload reverse complement commute on 2048 of 2048 states of those 8 charts and on 0 of the other 16 | Antipode commutation |
| 359 | **Codon space** GF(2)^6 with 64 states; triplet minimality from 16 < 21 <= 64 (rank ladder 2, 4, 16, 64 vs 21 semantic labels) | Rank ladder |
| 360 | **Four-mer as order-3 de Bruijn edge**: 256 directed edges on 64 codon vertices, compiles to one hQVM byte; the family sheet is the edge context | Byte compile |
| 361 | **Codon stage anatomy**: wobble bits 0,1 at UNA/ONA, middle bits 2,3 at BU/BU, first bits 4,5 at ONA/UNA (transverse cut of the stage palindrome) | Bit-stage map |
| 362 | **Codon reverse complement factorization** RC(q) = R_block(q) XOR delta_WC with R_block universal across all 24 charts | RC factorization |
| 363 | **Kinematic vs payload reverse complement**: agree on 0 of 256 bytes; residual confined to the rank-4 subspace W = {x1 = x3, x2 = x4} on all 24 encodings | Residual subspace |
| 364 | **Payload reverse complement** equals the chirality pole map (shell s to 6 - s) on all 256 pair-inversion bytes; Theta factors as S then F then R_block; signature residual depends only on length mod 4 | Pole map identity |
| 365 | **Chargaff's second parity rule** as pole invariance of the mononucleotide measure under antipodal exchange | Pole invariance |
| 366 | **GC skew decomposition on E. coli**: R^2 = 0.9941 over 1133 windows of 4096 bp on payload parity, family parity, and W membership; W residual near invariant across ori and ter | Skew regression |
| 367 | **Genomic depth-4 closure**: signature parity zero on 512,748 sliding frames (E. coli and yeast coding) | Parity census |
| 368 | **Compiled ORF signatures**: total-variation distance above 0.9 vs shuffle ensembles on three genomes; carrier-state marginals uniform after two averaged byte steps | ORF signature audit |
| 369 | **Genetic-code quotient**: fiber size profile 3 sextets, 5 quartets, 1 triplet, 9 pairs, 2 singletons plus 3 stops; meaning family-sheet invariant; length-1 signatures injective for 19 of 20 amino acids (serine excepted) | Fiber profile |
| 370 | **Affine hull ranks of fibers** (boxes rank 2, pairs rank 1, singletons rank 0, composites rank 3); serine the unique disconnected fiber (component ranks 1 and 2) | Hull ranks |
| 371 | **Synonymous cycle space** beta_1 = 27 = 8 x 3 + 1 + 1 + 1, equal to the Walsh layer j = 2 multiplicity 27 of H(3, 4) | Cycle rank |
| 372 | **Cycle-to-grade-2 projection**: rank 24 with 3-dimensional kernel and cokernel; four singular indicators (Leu bridge, Arg bridge, serine split, stop tree); serine and stop share one cokernel class | Projection audit |
| 373 | **Synonymous single-base edges** by codon position: 4 first / 1 middle (the stop path TAA to TGA) / 64 wobble; 64 of 69 total at wobble | Edge census |
| 374 | **Evolutionary singular-sector preference**: 54 NCBI reassignments over 13 codons, zero inside the five clean complete boxes (Pro, Thr, Val, Ala, Gly); uniform-targeting avoidance probability about 1.6e-9 | NCBI reassignment census |
| 375 | **Fold-plane direct sum** H = L_sense + P_fold: rank(L_sense) = 4, intersection {0}, L_sense equals the outer plane, on all 24 charts | Direct-sum check |
| 376 | **Hydropathy confirmation**: eta squared about 0.096 / 0.756 / 0.006 by codon position (Kyte-Doolittle); the middle base dominates on all 22 NCBI tables (0.583 to 0.772) | Hydropathy ANOVA |
| 377 | **tRNA identity census**: 18 of 30 curated elements in the anticodon, 2 at position 35 (fold plane); serine the unique multi-pole gauge-degenerate fiber matched by the anticodon-blind seryl-RS; the 5 direct-contact synthetase amino acids all single-pole injective on fold pole 00 | Identity census |
| 378 | **Genomic percolation ladder**: sense rank 4 reaches 256 states; sense plus stop or plus serine rank 5 reaches 1024; both rank 6 reach 4096 (square-root cluster theorem, see B9) | Reachability ladder |
| 379 | **Stop geometry**: {TAA, TAG, TGA, TGG} an affine rank-2 square with tryptophan completing it; stop leakage about 0.85; stop tree cycle rank 0 | Stop square |
| 380 | **Serine singularities**: unique fiber on two fold poles; unique reaching the complement horizon at ab-distance 12; unique length-1 signature collision under the family sheet flip | Serine audits |
| 381 | **Two code involutions**: the Rumer map sends all 8 complete boxes to incomplete; WC doublet complement preserves GC strength on 16 doublets and keeps 6 of 8 boxes complete; serine sits on both axes | Involution census |
| 382 | **Three K4 roles in the code**: family gauge sheet, wobble tolerance group (24 of 27 cycles), holonomy gates with fold exchanging W2 and W2' | K4 roles |
| 383 | **Ribosomal four-act advance**: Measure, Vary, Retrieve, Commit mapped to the byte acts at each codon step | Four-act map |
| 384 | **Subthermal occupation**: mean shells 2.854 / 2.890 / 2.897 (E. coli, yeast, chr22) below all 40 GC-matched nulls; QuBEC eta above all 12 null replicates; M2 below 4096 | Occupation audit |
| 385 | **Synonymous recoding** retains about three fifths of observed smoothness; two-layer decomposition of coding smoothness (code geometry vs native adjacency) | Recoding retention |
| 386 | **Adjacent-codon plaquette defects** centered at the kernel binomial value 3 (2.999, 3.002, 3.022 across the three genomes) | Plaquette mean |
| 387 | **Nested initiation events**: promoter -35 box family-sheet L1 about 0.50 vs null 0.10; start-codon window L1 about 0.24 vs 0.10; fold and reverse-complement channels at interior levels; two disjoint addresses about 40 bases apart | Initiation L1 |
| 388 | **Codon-pair bias radial decomposition**: chi-64 grouping eta squared 0.137 (E. coli) and 0.104 (yeast); shell share 0.042 and 0.053 (31 and 51 percent); monotonic profiles; 7 of 7 shells sign-consistent across both genomes; shell 0 bias +0.190 / +0.410 descending to shell 6 at -0.667 / -0.136; survives protein-fixed resampling | Radial bias |
| 389 | **Splice junction fold geometry**: mean fold disagreement 3.16 (donors) vs 2.15 (acceptors) over 8003 chr22 pairs; reverse complementation collapses the gap by about 30x | Splice fold |
| 390 | **Replichore conjugate holonomy**: matched defect density near 3, equal transport-weight invariants of 4, parity-zero circular product after even truncation; relative phi densities differ by about 0.25 percent (E. coli); W2 / W2' realization of palindromic conjugation | Replichore holonomy |
| 391 | **REBASE palindrome preference**: length congruent 2 mod 4 over 0 mod 4 among 3575 palindromic sites | REBASE census |
| 392 | **Aff_S6 classification**: order 46080 = 64 translations semidirect S_6; the labeled standard code a free orbit with trivial stabilizer; hard-constraint cluster 512 codes (1/90), connected | Aff_S6 orbit |
| 393 | **Hard-shell factorization**: 64 translations times the order-8 letter-bit swap group on pairs (0,1), (2,3), (4,5); absolute Leu/Arg/Ser placements 128/128/64 with 256 observed triples | Hard-shell factors |
| 394 | **Stop-boundary moduli**: 2240 admissible stop-plus-Trp boundaries in two Aff_S6 orbits (960, 1280); the standard code lies in the 960 minority orbit | Boundary moduli |
| 395 | **Local move classification**: 1159 weak one-codon moves collapse to 5 survivors (methionine expansions, one Aff_S6 orbit); pairwise combinations fail; 102 of 144 size-2 swaps pass | Local moves |
| 396 | **Variant wall breaches**: only 4 of 22 NCBI tables open the fold wall, each one edge, at poles 11 (TAG to TTG) and 01 (AAG to AGG); only tables 1 and 11 pass all 11 constraints | NCBI wall audit |
| 397 | **Equatorial capacity**: 20 = C(6, 3) matches the central shell multiplicity; 10 of 19 fiber direction sets align on the weight-3 shell | Equator capacity |
| 398 | **S6 edge characters** covariant under all 720 permutations in the natural form | Edge covariance |
| 399 | **Live-traffic identities**: equality-horizon occupancy exceeds GC nulls on 3 genomes; ab + horizon = 12 on measured means; same-amino-acid edge enrichment on 4 genomes | Live traffic |
| 400 | **Multi-layer genomic compile**: a 900-base E. coli window reports depth-4 parity-zero fraction 1, mean shell about 2.84, eta about 0.055, M2 about 4023, ab + horizon = 12 | Compile snapshot |

### B13. Receipt Geometry Analysis

| # | Feature | Method |
|---|---------|--------|
| 401 | **Compact receipts 16 to 20 bytes fit QR version 1 to 2**; a single payload bit flip fails seal, parity, and event | Tamper detection layers |
| 402 | **Receipt time field width 8 bytes**, one complete Z2 holonomy cycle (F^2 rest round trip) | Z2 cycle width identity |
| 403 | **Frame-aligned layouts L16/L20** keep the genealogy archive on stationary 4-byte depth-4 frames | Ledger frame hygiene |
| 404 | **Coordinate ledger storage about 1 byte per receipt** (depth delta + anchor); species-scale headroom large | Capacity and time-address audit |
| 405 | **FNV-1a name layer**: 10,000 sequential payloads, zero 64-bit collisions | Append-gate dispersion |
| 406 | **Kernel transport replay**: chi = chi_0 xor Q(word); the inverse restores rest; shell seal determinism | Trajectory verification |

### B14. Modal Logic and 3D Necessity

| # | Feature | Method |
|---|---------|--------|
| 407 | **Five foundational conditions** logically independent in the core modal system | Counterexample frames |
| 408 | **Consistency verified** via a three-world Kripke frame | Model construction |
| 409 | **3D necessity**: n = 3 is the unique dimension satisfying all 5 conditions | Lie-theoretic verification |
| 410 | **SE(3) = SU(2) rtimes R^3** forced by bi-gyrogroup consistency from ONA | Semidirect-product check |
| 411 | **sl(2) from BCH**: the depth-4 commutator forces the Lie algebra to close on 3 generators | Hall word exclusion |
| 412 | **Intelligence as BU closure** (preserve ancestry, identity, and individuality) | Operational definition |

---

*Last updated 2026-09-30 from the superintelligence test reports and specifications, the science analysis manuscripts, and the science repository experiment scripts.*