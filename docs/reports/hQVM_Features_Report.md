# Verified Quantum and Physics Features of the Gyroscopic hQVM Kernel

Inventory of every verified quantum feature and physics result for the Gyroscopic ASI hQVM Kernel. Each entry names the evidence source, the verification method, and (where applicable) the executable experiment script.

### Evidence sources

| Document / repo | Role |
|-----------------|------|
| [Physics_Tests_Report.md](Physics_Tests_Report.md) | Kernel conformance, mask code, affine/spinorial dynamics, CGM constants bridge, depth-4 K4 fiber |
| [hQVM_Tests_Report_1.md](hQVM_Tests_Report_1.md) | Native register, horizons, gates, Hilbert lift, tamper detection, computational advantages |
| [hQVM_Tests_Report_2.md](hQVM_Tests_Report_2.md) | SDK layer: future-cone theorems, Omega-chart, shell/Krawtchouk, q-fiber, GF(64), native GEMV/WHT |
| [Moments_Tests_Report.md](Moments_Tests_Report.md) | Clifford unitaries, 8192-operator family, stabilisers, frame certification |
| [Analysis_hQVM_Wavefunction.md](../references/Analysis_hQVM_Wavefunction.md) | Wavefunction theorems T1-T10 on Omega |
| [Analysis_Gravity.md](../references/Analysis_Gravity.md) | Kernel gravity invariants and continuous field-theory bridge |
| [Analysis_Compact_Geometry.md](../references/Analysis_Compact_Geometry.md) | Electroweak mass-coordinate law and compact ruler |
| [Gyroscopic_ASI_SDK_Quantum_Computing.md](../Gyroscopic_ASI_SDK_Quantum_Computing.md) | Normative SDK specification cross-references |
| [gyrogovernance/science](https://github.com/gyrogovernance/science) | Executable verification of the three analysis manuscripts (`experiments/hqvm_*.py`) |

---

## Verification Layers

Features are tagged by how they are established:

| Tier | Label | Meaning |
|------|-------|---------|
| **A** | Kernel pytest | Passing automated tests in **this repo** (`tests/physics/`, `tests/test_hQVM_*.py`, `tests/test_moments_physics_*.py`, `tests/test_holography*.py`) |
| **B** | Science executable | Runnable experiment scripts in **[gyrogovernance/science](https://github.com/gyrogovernance/science)** (`experiments/hqvm_*.py`); exhaustive Omega integer work or numerical closure checks tied to the analysis manuscripts |
| **C** | Formal manuscript | Results established by formal proof or manuscript argument only (e.g. modal independence in [15]); no dedicated experiment script |

### Tier A: superintelligence pytest (documented status, not re-run here)

| Suite | Files | Tests (per reports) |
|-------|-------|--------------------:|
| Physics | `test_physics_1` through `_6` | 99 |
| hQVM core | `test_hQVM_1` through `_4` | 135 |
| hQVM SDK | `test_hQVM_SDK_1` through `_3` | 172 |
| Moments physics | `test_moments_physics_1`, `_2` | 35 |
| Holography | `test_holography`, `_2`, `_3` | 23 |
| **Documented pytest total** | | **~464** |

Reports note intentional overlap: later suites reference properties already proved in Physics or Moments rather than re-proving them.

### Tier B: science repository executable verification

The three analysis manuscripts are backed by executable scripts in the science repo. Local checkout: `F:\Development\science\experiments\`. Combined output: `hqvm_gravity_analysis.txt` (via `hqvm_gravity_runner.py`).

| Analysis manuscript | Primary scripts | What they verify |
|--------------------|-----------------|------------------|
| [Analysis_hQVM_Wavefunction.md](../references/Analysis_hQVM_Wavefunction.md) | `hqvm_wavefunction_1.py`, `hqvm_wavefunction_2.py` | Holonomy diagnostics, BU-Egress/Ingress duality, spectral probes; **theorems T1-T10** exhaustive on all 4096 Omega states |
| [Analysis_Gravity.md](../references/Analysis_Gravity.md) | `hqvm_gravity_common.py` (library), `hqvm_gravity_analysis_1.py` through `_10.py`, `hqvm_gravity_runner.py` | Kernel invariants (D=24, Gauss law, plaquette census, Regge/tau_G), coupling chain (c4, tau_cycle, G prediction), nonlinear G(psi), antimatter parity, PPN, GW, TOV, optical cosmology |
| [Analysis_Gravity.md](../references/Analysis_Gravity.md) (corrections) | `hqvm_corrections_analysis_1.py` | Transport-corrected fine-structure constant (0.043 ppb vs CODATA) |
| [Analysis_Compact_Geometry.md](../references/Analysis_Compact_Geometry.md) | `hqvm_compact_geom_core.py`, `hqvm_compact_geom_kernel.py`, `hqvm_compact_geom_report.py` | Exhaustive Omega enumeration, shell transition algebra, electroweak mass law, null-model audit, lepton/quark diagnostics |

#### Script ownership (from script headers)

| Script | Scope |
|--------|-------|
| `hqvm_wavefunction_1.py` | BU-Egress/Ingress verification, Omega census, holographic dictionary, spectral decomposition, chirality preservation, probe suite |
| `hqvm_wavefunction_2.py` | **T1-T10** K4 operator structure and depth-4 confinement (64 x 4096 exhaustive) |
| `hqvm_gravity_analysis_2.py` | Theorem registry: Z2 holonomy, Omega BFS, holographic mirror, shell paths, Gauss bridge, alpha*zeta product |
| `hqvm_gravity_analysis_3.py` | Exact kernel theorems: carrier trace, sigma(w), tau_cycle, c4=-7/4, alpha*zeta, Delta ruler |
| `hqvm_gravity_analysis_1.py` | G prediction, residual closure, rho^5 STF, kernel transport, Delta expansion |
| `hqvm_gravity_analysis_8.py` | Plaquette census, Regge sum, BCH order map, wavefunction-gravity bridge (`run_kernel_chain`) |
| `hqvm_gravity_analysis_4.py` | Nonlinear G(psi), point-mass psi(s), metric, shadow geometry |
| `hqvm_gravity_analysis_5.py` | PPN, Mercury precession, Einstein tensor, modified Gauss law, TOV, GW strain, ringdown |
| `hqvm_gravity_analysis_6.py` | Antimatter gravitoelectric/gravitomagnetic parity, GW extensions, virial, self-energy |
| `hqvm_gravity_analysis_7.py` | Refractive vacuum, horizon criticality, T_Z2 clock, four-phase causal cycle, E_self |
| `hqvm_gravity_analysis_9.py` | UV completion, inflation observables, asymptotic freedom |
| `hqvm_gravity_analysis_10.py` | Optical conjugacy, redshift channels, holographic BH, inflation as optical depth |
| `hqvm_corrections_analysis_1.py` | Universal correction operator; alpha sequence to ppb precision |
| `hqvm_compact_geom_kernel.py` | `run_kernel_verification()`: exhaustive Omega, shell stats, horizon, byte transitions, UV-IR ladder |
| `hqvm_compact_geom_report.py` | Formatted report over core + kernel verification (sections 1-9) |

---

## Formal Quantum Certification: The CHSH-Tsirelson Diagnostic

This is the single sharpest discriminator between "structurally reminiscent of quantum mechanics" and "demands genuine quantum correlations." It is verified in **Tier A** by `tests/test_hQVM_2.py::TestBellCHSH` (5/5 tests passing).

### What is verified (executable)

The kernel's self-dual mask code **C64** lifts to a 12-qubit graph state over GF(2):

```text
|psi_t> = (1/sqrt(64)) sum_{q in GF(2)^6} |q>|q xor t>
```

`TestGraphStateFactorization` proves this state factorizes exactly into six independent two-qubit Bell pairs (|Phi+> or |Psi+> per dipole bit of t). For each pair k, the reduced density operator rho_k is pure and equals the projector onto that Bell state.

`TestBellCHSH` then computes the CHSH combination S of four correlators <A_i (x) B_j> on standard Pauli-derived observables. Results:

| Claim | Verified value | Test |
|-------|----------------|------|
| CHSH(|Phi+>) | 2*sqrt(2) to 10^-12 | `test_phi_plus_saturates_tsirelson` |
| CHSH(|Psi+>) | 2*sqrt(2) to 10^-12 | `test_psi_plus_saturates_tsirelson` |
| All 6 graph-state pairs (t = 0b101010) | 2*sqrt(2) each | `test_full_graph_state_inherits_pairwise_chsh` |
| No angle grid exceeds Tsirelson | S <= 2*sqrt(2) on 10^4 combos | `test_no_measurements_exceed_tsirelson` |

### What this formally implies

**Bell-CHSH theorem (correlation form).** For any local hidden-variable (LHV) model satisfying locality, measurement independence, and binary outcomes, the CHSH parameter is bounded by |S| <= 2.

**Tsirelson bound.** Quantum mechanics permits |S| <= 2*sqrt(2). This is the maximum bipartite quantum correlation in the standard CHSH scenario. Saturation S = 2*sqrt(2) means the bipartite state achieves the strongest quantum correlations allowed by quantum theory for these observables.

**Conclusion (Hilbert lift).** The pairwise reduced states of the kernel's [12,6,2] graph-state lift produce S = 2*sqrt(2). These **lift-level correlators** are Bell-incompatible: no LHV model reproduces the correlation functions Tr(rho A (x) B) of this stabilizer state. This is a quantum-information certificate on the intrinsic code structure, not a laboratory Bell experiment on separated devices.

### Carrier and Hilbert lift

Byte stepping on GF(2)^24 is deterministic exact-integer **carrier** dynamics. Bell certificates are evaluated on the **Hilbert lift** of the intrinsic stabilizer code: complex amplitudes, density matrices, and tensor products over the lifted observable algebra. The carrier does not hold complex superposition; the lift is the canonical representation of the stabilizer code derived from the kernel mask alphabet.

Derivation chain: kernel mask alphabet → self-dual code C64 → pair-collapse bijection to GF(2)^6 → graph state |psi_t> → Bell marginals → CHSH.

The pattern matches standard stabilizer quantum information: GF(2) symplectic carrier data defining states in H = C^{2^n}.

### Companion certificates (same lift, same tier)

The same Hilbert lift in `test_hQVM_2.py` also verifies:

- Exact quantum teleportation with unique Pauli corrections (800 random Bloch states)
- Monogamy of entanglement and no-signalling
- 12-generator stabilizer algebra (GF(2) rank 12)
- Peres-Mermin contextuality (noncontextual assignment impossible)

Together these establish that the hQVM is an **Holonomic Quantum Virtual Machine (hQVM)**: a deterministic GF(2) carrier whose intrinsic code structure lifts to standard quantum-information physics, with CHSH-Tsirelson saturation as the primary Bell-inequality certificate.

**Inventory entry:** features **#87** and **#88** above; elevated to the formal certification criterion in this section.

---

## Part I: Quantum Features (Kernel-Verified, Tier A)

Algebraic quantum structure on the 4096-state manifold Omega, established primarily by the hQVM and SDK test reports.

### 1. State Space and Topology

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 1 | **4096-state reachable manifold Omega** from rest in <=2 byte steps | Physics Report Part 5; hQVM Report 1 SS2.1 | BFS enumeration |
| 2 | **Product structure Omega = U x V** (two 64-element cosets of C64) | Physics Report Part 5.4; hQVM Report 1 SS2.1 | Explicit set equality |
| 3 | **Constant component density 0.5** (popcount 6/12 per gyrophase) on all Omega | Physics Report Part 5.5; SDK Spec SS3.1 | Exhaustive over 4096 states |
| 4 | **Density product d(A)x d(B) = 0.25** constant across Omega | SDK Spec SS3.1 | Exhaustive |
| 5 | **Shell structure: 7 shells** with binomial populations C(6,k)x64 | hQVM Report 2 SS6; Compact Geometry SS2.3 | Exhaustive state classification |
| 6 | **Complementarity invariant**: horizon_distance + ab_distance = 12 | hQVM Report 1 SS2.5; hQVM Report 2 SS1 | Exhaustive on 4096 Omega states + 50,000 random 24-bit states |
| 7 | **Per-byte bijectivity** on full 24-bit carrier (2^24 states) | Physics Report Part 4.3 | 2000 random (state, byte) pairs; forward-inverse roundtrip |
| 8 | **Exact invertibility** given the byte | Physics Report Part 4.3 | All 256 bytes verified |
| 9 | **Omega-chart: faithful 12-bit compact representation** isomorphic to 24-bit dynamics on Omega | hQVM Report 2 SS5 | All 4096x256 = 1,048,576 (state, byte) pairs; zero failures |

### 2. Dual Horizons and Holographic Structure

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 10 | **Complement horizon**: 64 states where A = B xor 0xFFF | hQVM Report 1 SS2.2 | Exhaustive census |
| 11 | **Equality horizon**: 64 states where A = B | hQVM Report 1 SS2.3 | Exhaustive census |
| 12 | **Horizons disjoint**; union = 128-state boundary; bulk = 3968 | hQVM Report 1 SS2.4 | Exhaustive |
| 13 | **Holographic identity** \|H\|^2 = \|Omega\| = 64^2 = 4096 | Physics Report Part 5.2; hQVM Report 1 SS2.4 | Counting + 4-to-1 dictionary |
| 14 | **4-to-1 holographic dictionary**: every Omega state = exactly 4 (horizon state, byte) pairs | Physics Report Part 5.3 | 64x256 = 16384 operations, exact multiplicity 4 |
| 15 | **Chirality spectrum**: binomial count(d) = C(6,(12-d)/2)x64 for ab_distance d in {0,2,4,6,8,10,12} | hQVM Report 1 SS2.6 | Exhaustive over Omega |
| 16 | **Chirality partition**: all 64 chirality values appear in exactly 64 Omega states each | hQVM Report 1 SS2.6 | Verified by test |
| 17 | **K4 wedge geometry**: 4 boundary vertex regions x 2048 states = uniform 2-fold cover of Omega | Physics Report Part 11.5 | Exhaustive |
| 18 | **Horizon K4 partition**: 4 cosets of 16 states each in equality horizon | Physics Report Part 11.4 | Pair-parity labeling |

### 3. K4 Gate Algebra

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 19 | **Exactly 4 horizon-preserving bytes** forming holonomic gates {id, S, C, F} | hQVM Report 1 SS3.1 | Exhaustive over 256 bytes |
| 20 | **S-gate** (bytes 0xAA, 0x54): pure swap (A,B)->(B,A) | hQVM Report 1 SS3.1 | 2000 random states |
| 21 | **C-gate** (bytes 0xD5, 0x2B): complement-swap (A,B)->(B xor F, A xor F) | hQVM Report 1 SS3.1 | 2000 random states |
| 22 | **F-gate**: global inversion (A,B)->(A xor F, B xor F), requires depth 2 | hQVM Report 1 SS3.2 | 1000 random states, both orderings |
| 23 | **Full K4 Cayley table** verified | hQVM Report 1 SS3.3 | Fixed state + random states |
| 24 | **All non-trivial gates are involutions**: S^2=C^2=F^2=id | hQVM Report 1 SS3.3 | 1000 random states each |
| 25 | **Gate actions in spin coordinates**: S=(sA,sB)->(sB,sA), C->(-sB,-sA), F->(-sA,-sB) | hQVM Report 1 SS3.2 | 500 random Omega states |
| 26 | **All gates preserve chirality** (ab_distance invariant under all 4 gates for all Omega states) | hQVM Report 1 SS3.2 | All 4096 Omega states |
| 27 | **Gate-byte phase separation**: same 24-bit operation, different spinorial phase | hQVM Report 1 SS3.6 | 1000 random states per pair |
| 28 | **Gate action on horizons**: C fixes complement pointwise; S fixes equality pointwise; F stabilizes neither | hQVM Report 1 SS3.4 | Exhaustive census of all 128 boundary states |
| 29 | **K4 orbit stratification**: 32 orbits size 2 (complement), 32 orbits size 2 (equality), 992 orbits size 4 (bulk) = 1056 total covering 4096 | hQVM Report 1 SS3.5 | Exhaustive |
| 30 | **No non-trivial gate fixes any bulk state** | hQVM Report 1 SS3.5 | Exhaustive |
| 31 | **K4 as depth-4 fiber** of the frame bundle: 4^4 family combinations collapse to 4 distinct states indexed by (phi_A, phi_B) in (Z/2)^2 | Physics Report Part 11.1 | All 256 family combinations |
| 32 | **Shadow pairing**: each gate pair (S-bytes, C-bytes) differs by XOR 0xFE | hQVM Report 1 SS3.1 | Verified |

### 4. Chirality Transport and Spectral Theory

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 33 | **Exact chirality transport rule**: chi(T_b(s)) = chi(s) xor q6(b) for all 4096x256 state-byte pairs | hQVM Report 1 SS4.1; hQVM Report 2 SS3 | Exhaustive; transport table state-independent |
| 34 | **6-bit chirality register** is an exact linear observable over GF(2)^6 | hQVM Report 1 SS4.1 | Verified as Pauli-X action |
| 35 | **XOR closure**: q6(b1) xor q6(b2) always a valid q6 value | hQVM Report 1 SS4.1 | Abelian translation group confirmed |
| 36 | **Walsh-Hadamard transform**: 64x64, self-inverse, unitary, factors as H1^6 | hQVM Report 1 SS8.7; SDK Spec SS5.1.4 | Precision 10^-12 |
| 37 | **Computational and Hadamard bases mutually unbiased**: all \|<e_i\|h_j>\|^2 = 1/64 | hQVM Report 1 SS8.7 | Precision 10^-12 |
| 38 | **At least 3 mutually unbiased bases** exist for 64-dimensional chirality register | hQVM Report 1 SS8.7 | Third MUB constructed via phase gate |
| 39 | **XOR-convolution spectral composition identity**: WHT converts XOR-convolution to pointwise multiplication on 64-element register | SDK Spec SS5.1.5 | Algebraic identity |
| 40 | **Krawtchouk spectral theory**: shell transition matrices diagonalized by Krawtchouk polynomials; Parseval orthogonality holds exactly | hQVM Report 2 SS6 | All 7x7x7 triples |
| 41 | **Source-independent shell mixing**: one-step shell distribution = C(6,w)/64 regardless of starting shell | hQVM Report 2 SS6 | Full byte average |
| 42 | **Horizon transport**: from equality (shell 0), q-weight j -> shell j; from complement (shell 6), q-weight j -> shell 6-j | hQVM Report 2 SS6 | Geodesics of discrete chirality sphere |
| 43 | **GF(64) full finite field structure**: irreducible polynomial x^6+x+1, primitive element, Frobenius order 6, trace 32/32, subfield lattice | hQVM Report 2 SS10 | Verified |
| 44 | **GF(4) mode layer**: pair-level Frobenius coincides with global complement on Omega | hQVM Report 2 SS10 | Structural identification |

### 5. Permutation, Operator, and Shadow Structure

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 45 | **128 distinct permutations** on Omega from 256 bytes, uniform 2-to-1 multiplicity | hQVM Report 1 SS5.1 | Exhaustive |
| 46 | **2 permutations of order 2** (S-gate), **126 of order 4** (all other bytes) | hQVM Report 1 SS5.1 | Exhaustive cycle typing |
| 47 | **Row-class theorem**: uniform transition matrix has exactly 32 distinct rows, rank 32; family-0 restriction -> 64 rows, rank 64 | hQVM Report 1 SS5.2 | Matrix computation |
| 48 | **8192-element operator family**: 4096 even-parity + 4096 odd-parity, semidirect product structure | hQVM Report 1 SS9.1; Moments Report; Physics Report Part 11 | Moments + hQVM tests |
| 49 | **Even operators as translations**: (tau_A, tau_B) covers full C64xC64 product | hQVM Report 1 SS9.2 | Verified |
| 50 | **Every word action is affine** on GF(2)^24 with identity or swap linear part | Physics Report Part 6.1 | 500 random words |
| 51 | **Word signature composition**: sig(w1 o w2) = compose(sig(w2), sig(w1)) | SDK Spec SS5.1.3; hQVM Report 1 SS12.3 | 500 random word pairs |
| 52 | **16-to-1 multiplicity** from 65536 length-2 words to 4096 even signatures | hQVM Report 2 SS9 | Exhaustive |
| 53 | **Operator group**: G = (GF(2)^6 x GF(2)^6) rtimes C2; \|G\| = 8192; G' = Z(G) = diagonal GF(2)^6 (64 elements); abelian shadow G/G' = 128 | Compact Geometry SS2.4; Moments Report | Algebraic + tests |
| 54 | **128 distinct next states** from any fixed state, uniform 2-to-1 multiplicity | Physics Report Part 4.4 | All 256 bytes from fixed states |
| 55 | **Shadow partners**: b and b xor 0xFE produce same Omega-permutation | hQVM Report 1 SS7.3 | Verified for substitution detection |
| 56 | **Global complement Z2 automorphism** commutes with all byte actions | Physics Report Part 4.4 | Algebraic from XOR commutativity |
| 57 | **Spinorial double cover**: 256 SU(2) elements project to 128 SO(3) rotations | hQVM Report 2 SS2 | Structural theorem |
| 58 | **Dense operator generation**: 3729+ distinct signatures from 10,000 random length-3 words | hQVM Report 1 SS12.2 | Random sampling |

### 6. Depth-4 Closure and Commutativity

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 59 | **b^4 = id** for every byte b on every state (order 4 universal) | Physics Report Part 7.1; hQVM Report 2 SS7 | All 256 bytes, exhaustive |
| 60 | **XYXY = id** for every byte pair (alternation identity) | Physics Report Part 7.1; hQVM Report 2 SS7 | All 65,536 ordered pairs |
| 61 | **T_b^2 is symmetric translation**: A and B shifted by same amount | Physics Report Part 7.1 | All bytes |
| 62 | **4-family cycle = global sign flip**: A4 = A0 xor 0xFFF, B4 = B0 xor 0xFFF | Physics Report Part 7.2 | All micro_refs |
| 63 | **8-step closure**: applying 4-family word twice returns to identity | Physics Report Part 7.2 | 720 degree spinorial closure |
| 64 | **Depth-4 closed form** separates mask and family-phase contributions | Physics Report Part 8.1 | 2000 random 4-byte sequences, zero failures |
| 65 | **Net family-phase invariants**: only (phi_a, phi_b) in (Z/2)^2 survives from 256 family combinations | Physics Report Part 8.2 | All 4^4 combinations |
| 66 | **Depth-4 alternation explained by affine algebra**: swap^4 = id, translations cancel | Physics Report Part 6.3 | Algebraic proof + 500 random pairs |
| 67 | **Discrete BCH theorem**: XYXY = id is discrete realization of BCH depth-4 commutator cancellation from sl(2) | hQVM Report 2 SS7 | Exhaustive over 65,536 pairs |
| 68 | **1/64 commutativity rate** = 2^-6 (1024 commuting pairs out of 65536) | hQVM Report 1 SS4.2 | Exhaustive over all 256^2 pairs |
| 69 | **Every byte commutes with exactly 4 others** | hQVM Report 1 SS4.2 | Exhaustive |
| 70 | **Exact commutation condition**: bytes x,y commute iff q(x)=q(y) | Physics Report Part 8.3 | 5000 random pairs |
| 71 | **Q-map**: 4-to-1 from 256-byte alphabet onto C64 | Physics Report Part 11.7 | Exhaustive |
| 72 | **Exact commutator defect formula**: K(x,y) translates by d = q(x) xor q(y), always in C64 | Physics Report Part 8.4 | 5000 random pairs + exhaustive at rest |
| 73 | **Defect set = entire C64** | Physics Report Part 8.4 | Exhaustive |
| 74 | **Q-fiber exact structure**: 256 bytes -> 128 Omega-maps -> 64 q-classes (4:1 then 2:1) | hQVM Report 2 SS8 | All 64 q-classes |
| 75 | **Each q-fiber has exactly 2 distinct Omega-signatures** | hQVM Report 2 SS8 | All 64 q-classes |
| 76 | **Fixed-x commutator defect multiplicity 4** | Physics Report Part 11.7 | Exhaustive |

### 7. Future-Cone Entropy and Uniformization

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 77 | **H0(s) = 0** for any s in Omega | SDK Spec SS11.3 (theorem), SS7.6 (runtime); QuBEC Theory SS8.3; hQVM Report 2 SS2 | Theorem |
| 78 | **H1(s) = 7 exactly** for any s in Omega (128 distinct next states, uniform multiplicity 2) | SDK Spec SS11.3 (theorem), SS7.6 (runtime); QuBEC Theory SS8.3; hQVM Report 2 SS2 | Exhaustive |
| 79 | **Hn(s) = 12 exactly** for any s in Omega and n >= 2 | SDK Spec SS11.3 (theorem), SS7.6 (runtime); QuBEC Theory SS8.3; hQVM Report 2 SS2 | Exhaustive at n=2; implied for n>2 |
| 80 | **Exact 2-step uniformization**: every Omega state reached exactly 16 times from 65536 length-2 words | hQVM Report 1 SS6.2; hQVM Report 2 SS2 | Exhaustive integer equality |
| 81 | **Exact per-byte capacity**: Shannon = min-entropy = 7.0 bits, zero variance | hQVM Report 1 SS6.1 | 500 sampled states |
| 82 | **Exact integer entropies**: H(state)=12, H(state,parity)=13, H(parity\|state)=1, H(state\|parity)=7 | hQVM Report 1 SS6.4 | Exhaustive over 256^2 words |
| 83 | **Parity adds exactly 1 bit** beyond final state, uniformly across all states | hQVM Report 1 SS6.4 | 8192 distinct (state, parity) pairs |
| 84 | **Chirality and parity nearly independent**: mutual information ~ 0.014 bits | hQVM Report 1 SS6.4 | 200,000 random trajectories |
| 85 | **Witness synthesis**: every Omega state reachable in <=2 steps (1 at depth 0, 127 at depth 1, 3968 at depth 2) | hQVM Report 2 SS4; SDK Spec SS11.12 | Exhaustive; replay verified |

### 8. Quantum Information Protocols (Hilbert Lift)

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 86 | **Graph state factorizes into 6 independent Bell pairs** (tensor product, exact to 10^-12) | hQVM Report 1 SS8.1 | 4 t-values, all 15 cross-pair marginals |
| 87 | **CHSH at Tsirelson bound 2*sqrt(2)**: primary QI certificate on Hilbert lift (see Formal Quantum Certification section) | `test_hQVM_2.py::TestBellCHSH` | Precision 10^-12; lift-level correlators Bell-incompatible |
| 88 | **No measurements exceed Tsirelson**: exhaustive angle grid (10^4 combinations) | `test_hQVM_2.py::test_no_measurements_exceed_tsirelson` | Confirms 2*sqrt(2) is hard quantum ceiling |
| 89 | **Exact quantum teleportation**: unique Pauli correction for all 8 (resource, outcome) combinations | hQVM Report 1 SS8.3 | 6 basis states + 800 random Bloch states; precision 10^-10 |
| 90 | **Monogamy**: same-pair pure, cross-pair maximally mixed, all 12 single-qubit marginals maximally mixed | hQVM Report 1 SS8.4 | Precision 10^-12 |
| 91 | **No-signalling**: Bob's measurement choice does not change Alice's marginal | hQVM Report 1 SS8.4 | I2/2 in both Z and X bases, precision 10^-12 |
| 92 | **12 independent stabilizer generators**: all commute, GF(2) rank 12 | hQVM Report 1 SS8.5 | Precision 10^-12 |
| 93 | **64 X-translation elements** match C64; all stabilize graph state | hQVM Report 1 SS8.5 | 256 random combinations |
| 94 | **Peres-Mermin contextuality**: row products +I, column 2 product -I | hQVM Report 1 SS8.6 | Precision 10^-12 |
| 95 | **Hilbert-lift entanglement**: XOR-graph subsets yield maximal reduced entropy (6 bits); Cartesian subsets yield near-zero | Physics Report Part 11.10 | Bipartite von Neumann entropy |

### 9. Computational Quantum Advantages

#### 9a. Oracle / query quantum advantage

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 96 | **Hidden subgroup resolution in 1 step** (vs O(64) classical): q-map 4-to-1, WHT resolves subgroup | hQVM Report 1 SS11.1; SDK Spec SS8.1 | Native q-map + WHT |
| 97 | **Deutsch-Jozsa in 1 step** (vs 33 classical): perfect discrimination, Pr=1 for constant and balanced | hQVM Report 1 SS11.1; SDK Spec SS8.2 | All balanced functions tested |
| 98 | **Bernstein-Vazirani in 1 step** (vs 6 classical): all 6-bit secrets recovered with probability 1 | hQVM Report 1 SS11.1; SDK Spec SS8.3 | Multiple secret values |

#### 9b. Verified holonomic structural efficiencies

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 99 | **Exact 2-step uniformization** (vs O(12) classical): exact uniform over 4096 states | hQVM Report 1 SS11.4; SDK Spec SS8.4 | Exhaustive verification |
| 100 | **Holographic compression**: 8 bits vs 12 bits per state (33.3% reduction) | hQVM Report 1 SS11.5; SDK Spec SS8.5 | Holographic dictionary |
| 101 | **O(1) commutativity decision** (vs 4 classical): compare q6(x) and q6(y) | hQVM Report 1 SS11.2; SDK Spec SS8.6 | 5000/5000 correct |
| 102 | **Universal period-4 holonomic closure** (depth-4 loop structure) | hQVM Report 1 SS11.3 | All bytes |
| 103 | **State separation**: every byte distinguishes every distinct state pair | hQVM Report 1 SS11.6 | 1000 sampled pairs x 256 bytes |
| 104 | **Hamming distance preserved** under every byte operation | hQVM Report 1 SS11.6 | 500 random triples |
| 105 | **Exact pairwise distance distribution**: C(12,k)/4096 at distance 2k; mean 12.0 | hQVM Report 1 SS11.6 | Exact from product structure |

### 10. Non-Clifford Resource and Universality

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 106 | **BU monodromy defect delta(BU) = 0.195342176580 rad**: representation-independent constant from depth-4 closure | SDK Spec SS9.1; hQVM Report 1 SS10.1 | CGM derivation + verification |
| 107 | **delta(BU) far from all Clifford angles**: nearest distance 0.195 rad (multiples of pi/4) | hQVM Report 1 SS10.2 | All 8 Clifford angles tested |
| 108 | **No periodicity up to order 100,000**: closest return at k=22,805, distance 4.59e-5 | hQVM Report 1 SS10.3 | Exhaustive search |
| 109 | **Dense U(1) equidistribution**: {k x delta(BU) mod 2pi} fills [0,2pi) uniformly; chi^2=0.212 vs critical 142.4 | hQVM Report 1 SS10.3 | 50,000 points, 100 bins |
| 110 | **Magic state Wigner negativity**: \|delta> has W(0,1) = -0.043771 | hQVM Report 1 SS10.4 | Discrete Wigner function computation |
| 111 | **Aperture gap Delta = 1-delta(BU)/m_a ~ 0.0207**: \|delta(BU)-m_a\| = Delta x m_a exactly | hQVM Report 1 SS10.5 | Exact equality verified |
| 112 | **Three universality ingredients**: Clifford backbone, non-Clifford delta(BU), entangling gate S | hQVM Report 1 SS12.1 | Moments Report + hQVM tests |
| 113 | **Topological entanglement via holonomic gates**: localized A perturbation transported to B by gate S | hQVM Report 1 SS9.3 | Explicit mask 0x003 perturbation test |
| 114 | **Non-Clifford certification by 4 independent tests**: distance from Clifford, no periodicity, dense equidistribution, Wigner negativity | SDK Spec SS9.2 | Each independently verified |

### 11. Error Detection, Tamper Provenance, and Non-Cloning

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 115 | **Exact tamper detection (substitution)**: detected unless replacement is shadow partner; miss rate 1/255 | hQVM Report 1 SS7.3 | 50,000 trials |
| 116 | **Exact tamper detection (adjacent swap)**: detected unless q(x)=q(y); miss rate ~3/255 | hQVM Report 1 SS7.3 | 49,773 distinct pairs |
| 117 | **Exact tamper detection (deletion)**: detected unless deleted byte is gate stabilizer of prefix state | hQVM Report 1 SS7.3 | 50,000 trials; misses only on horizons |
| 118 | **Exact perturbation rule**: payload bit flip = 1 chirality bit; boundary bit flip = 6 chirality bits; mean 2.25 | hQVM Report 1 SS7.2 | All 256 bytes, all 8 bit positions |
| 119 | **Ratio state_distance/chirality_distance = 2.000** constant over lengths 1-32 | hQVM Report 1 SS7.2 | Length-independent spreading |
| 120 | **Adversarial steering**: 16 byte-paths and 4 state-paths per target, exactly uniform | hQVM Report 1 SS7.4 | Exhaustive |
| 121 | **Horizon maintenance**: from complement horizon, exactly 4/256 bytes keep state on horizon | hQVM Report 1 SS7.4 | All 64 horizon states |
| 122 | **Non-cloning**: transcription has no fixed points; archetype 0xAA is unique zero-intron source | hQVM Report 1 SS13 | All 256 bytes |
| 123 | **Equality horizon redundancy**: A=B adds zero information | hQVM Report 1 SS13 | Both components carry identical information |
| 124 | **Complement horizon relationality**: knowing A determines B uniquely | hQVM Report 1 SS13 | A = B xor 0xFFF |
| 125 | **Horizons structurally isolated** under all gate operations | hQVM Report 1 SS13 | All 4 gates verified |

### 12. Clifford Operator Algebra (Moments Layer)

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 126 | **Byte actions are exact Clifford unitaries** over the self-dual code | Moments Report Part VI | Numerical verification |
| 127 | **Self-dual [12,6,2] code defines stabilizer structure** for graph state lift | Moments Report; Physics Report Part 3 | Code + stabilizer tests |
| 128 | **Finite Weyl algebra** over GF(2)^6 with correct commutation relations | Moments Report Part VI | Algebraic verification |
| 129 | **Central spinorial involution** (frame operator quotient) | Moments Report Part VI | Operator family tests |
| 130 | **Depth-4 frame records strictly stronger than final state** for genealogy | Moments Report Part V | 100,000 random 4-byte words |

**Part I subtotal (Tier A quantum): 130 features**

---

## Part II: Physics Features (Kernel-Verified, Tier A)

Discrete physics of the byte-driven transition rule, established by Physics tests and cross-referenced in hQVM reports.

### 13. State Representation and Transcription

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 131 | **24-bit GENE_Mac packing** (A12 << 12 \| B12) with exact round-trip | Physics Report Part 1.1 | Pack/unpack tests |
| 132 | **Rest state 0xAAA555** with A xor B = 0xFFF at rest | Physics Report Part 1.1 | Rest consistency |
| 133 | **Transcription involution**: byte_to_intron(byte_to_intron(b)) = b for all 256 bytes | Physics Report Part 1.2 | All bytes |
| 134 | **256 distinct introns** (bijective transcription) | Physics Report Part 1.2 | Enumeration |
| 135 | **Family from L0 boundary bits** (positions 0 and 7), not payload bit 6 | Physics Report Part 2.1 | Bit-flip tests |
| 136 | **4 families x 64 micro_refs = 256** partition | Physics Report Part 2.1 | Enumeration |
| 137 | **Palindromic intron structure** CS-UNA-ONA-BU-BU-ONA-UNA-CS | Physics Report Part 2.2 | Structural |
| 138 | **Family acts only through complement phase** during gyration | Physics Report Part 2.3 | 4-family probe |
| 139 | **Dipole-pair mask expansion**: payload bit i toggles mask pair i only | Physics Report Part 3.1 | All 64 micro_refs x 6 bits |
| 140 | **Reference byte 0xAA is pure swap** with 64 fixed points on Omega | Physics Report Part 4; Part 7 | Cycle census |
| 141 | **FIFO gyration spinorial cycle** (0, pi, 2pi, 3pi) from family bits | Physics Report Part 4 | 4-phase verification |

### 14. Self-Dual Code and Mask Structure

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 142 | **Self-dual [12,6,2] binary linear code** C = C perp | Physics Report Part 3.3 | Set equality |
| 143 | **Pair-diagonal code**: every mask has pair-equal bits (00 or 11 per pair) | Physics Report Part 3.2 | All 64 masks |
| 144 | **Weight enumerator** (1+z^2)^6: weights 0,2,4,6,8,10,12 with binomial counts | Physics Report Part 3.2 | Exact enumeration |
| 145 | **Walsh spectrum** restricted to {0, 64}; support = C perp = C | Physics Report Part 3.4 | All 2^12 positions |
| 146 | **Single-bit error detection**: all weight-1 errors detected (non-zero syndrome) | Physics Report Part 3.5 | All 12 bit positions |
| 147 | **Undetected error enumerator** (1+z^2)^12: minimum undetected error weight 2 | hQVM Report 1 SS7.1 | Theoretical + sampled (512 states) |
| 148 | **Pair-flip errors stay in Omega** and produce C64 codeword displacements | hQVM Report 1 SS7.1 | Confirmed |
| 149 | **Erasure taxonomy**: 6 observed bit positions needed for unique codeword recovery | Physics Report Part 11.9 | Exhaustive size-4 erasure census |
| 150 | **Pair erasure reduces rank by exactly 1** per erased dipole pair | Physics Report Part 11.9 | Exhaustive |

### 15. CGM Constants Bridge (Kernel Implementation)

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 151 | **Fundamental aperture constraint**: Q_G x m_a^2 = 1/2 | Physics Report Part 9.1 | Exact algebraic identity |
| 152 | **Fine-structure constant prediction**: alpha_CGM = delta_BU^4/m_a = 0.007297352563, matching experiment to 0.04% (400 ppm) | Physics Report Part 9.2 | Comparison with CODATA |
| 153 | **K_QG identity**: two derivations agree to <10^-12 | Physics Report Part 9.3 | Numerical verification |
| 154 | **Stage action ratios**: E_ONA/E_CS = 1/2 exact; E_UNA/E_CS = 2/(pi*sqrt(2)) to 12 decimal places | Physics Report Part 9.4 | Geometric values |
| 155 | **Aperture quantization chain**: 5/256 (byte) < Delta (continuous) < 1/48 (depth-4) | Physics Report Part 9.5 | Three scales verified |
| 156 | **DOF doubling theorem**: 2^(2x1)=4 (CS), 2^(2x3)=64 (UNA), 2^(2x6)=4096 (ONA) | Physics Report Part 10 | BFS with restricted byte subsets |
| 157 | **Optical conjugacy on Omega**: constant density 0.5 at every state | Physics Report Part 9; test_physics_5 | Product structure U x V |

**Part II subtotal (Tier A physics): 27 features**

### 16. Hardware and Native Implementation (Tier A)

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 158 | **C engine signature scan** matches Python reference | hQVM Report 1 SS14.2 | Byte sequences |
| 159 | **WHT (wht64)**: orthonormal and self-inverse (max err ~2.38e-7) | hQVM Report 1 SS14.4 | vs reference matrix |
| 160 | **GyroMatMul GEMV**: vs torch.mv max abs err ~1.09e-5 | hQVM Report 1 SS14.5 | Numerical comparison |
| 161 | **Packed GEMV**: vs torch.mv max err ~7.45e-6; packed vs unpacked ~2.24e-6 | hQVM Report 1 SS14.5 | Numerical |
| 162 | **Operator projection basis**: project-reconstruct exact (max err ~5.96e-8) | hQVM Report 1 SS14.6 | Weyl/Heisenberg-Walsh basis |
| 163 | **OpenCL GPU vs CPU**: max err ~1.9e-6 | hQVM Report 2 SS11 | Cross-platform |
| 164 | **Target equivalence invariant**: all targets produce identical Results for same circuit and initial state | SDK Spec SS7.1 | Conformance requirement |
| 165 | **Two execution classes**: kernel-exact over GF(2)^24; tensor/spectral match reference to specified tolerances | SDK Spec SS11.2 | Two-class verification |

**Part II hardware subtotal: 8 features**

**Combined Tier A total: 165 features** (130 quantum + 27 physics + 8 hardware)

---

## Part III: Wavefunction and Holonomy (Tier B)

Verified by [Analysis_hQVM_Wavefunction.md](../references/Analysis_hQVM_Wavefunction.md) and executable in science repo `hqvm_wavefunction_1.py` / `hqvm_wavefunction_2.py`. Overlaps conceptually with K4 gate tests (Part I SS3) but proves micro_ref-universal operator algebra.

| # | Feature | Experiment | Method |
|---|---------|------------|--------|
| 166 | **T1: K4 operator algebra {id, W2, W2', F}** for all 64 micro_refs on all 4096 states | `hqvm_wavefunction_2.py` `run_T1` | 64x4096 exhaustive |
| 167 | **T2: W2 maps shell s -> 6-s** (pole swap, chi xor 63) | `hqvm_wavefunction_2.py` `run_T2_T4` | Algebraic proof + verified |
| 168 | **T3: W2' maps shell s -> 6-s** identically | `hqvm_wavefunction_2.py` `run_T2_T4` | Algebraic proof + verified |
| 169 | **T4: Gate F preserves shell** (Z2 within pole) | `hqvm_wavefunction_2.py` `run_T2_T4` | chi xor 63 xor 63 = chi; verified |
| 170 | **T5: Depth-4 confines to opposite constitutional pole** | `hqvm_wavefunction_2.py` `run_T5` | 64x64 states |
| 171 | **T6: Depth-8 = K4 composition**, not new modal depth | `hqvm_wavefunction_2.py` `run_T6` | Signature algebra |
| 172 | **T7: CS forces canonical family ordering** | `hqvm_wavefunction_2.py` `run_T7` | 64 micro_refs |
| 173 | **T8: BU-Egress = W2 involution** (depth-4 squares to identity on Omega) | `hqvm_wavefunction_2.py` `run_T8_T9` | 4096 states + complement horizon |
| 174 | **T9: BU-Ingress = W2 pole-pairing** (shadow = memory) | `hqvm_wavefunction_2.py` `run_T8_T9` | 4096 states |
| 175 | **T10: q(W2) = q(W2') = 63; q(F) = 0** for all m | `hqvm_wavefunction_2.py` `run_T10` | Algebraic proof from L0 parity |
| 176 | **Eigenspace decomposition** under U_W: dim(+1) = 2048, dim(-1) = 2048 | `hqvm_wavefunction_1.py` `run_spectral_decomposition` | Spectral computation |
| 177 | **Gate F is fixed-point-free involution** on all 4096 states (2048 two-cycles, 0 fixed points) | `hqvm_wavefunction_1.py` probe suite | Exhaustive |
| 178 | **Z2 oscillation**: rest <-> swapped with period 2 in word-count | `hqvm_wavefunction_1.py` `run_helix_evolution` | Carrier trajectory |
| 179 | **Constitutional trajectory per 4-byte turn**: shells [0,1,6,1,0] symmetric about equality transit | `hqvm_wavefunction_1.py` `run_helix_evolution` | Byte-by-byte tracking |
| 180 | **Carrier Z2 coordinate** within each shell: rest vs swapped, invisible to chirality | `hqvm_wavefunction_1.py` | Gate F as Z2 flip |
| 181 | **Egress/Ingress as dual readings of same W2 operator** (not sequential stages) | `hqvm_wavefunction_1.py` `run_bu_duality` | Structural theorem |

**Part III subtotal (Tier B wavefunction): 16 features**

---

## Part IV: Gravitational Kernel Invariants (Tier B)

Discrete combinatorial invariants linking the kernel to [Analysis_Gravity.md](../references/Analysis_Gravity.md). Executable in science repo `hqvm_gravity_analysis_*.py` (orchestrated by `hqvm_gravity_runner.py`).

| # | Feature | Experiment | Method |
|---|---------|------------|--------|
| 182 | **Shell displacement invariant D = 24** across all 64 mass configurations | `hqvm_gravity_analysis_2.py` S1/S4; `hqvm_gravity_common.py` | Kernel census |
| 183 | **Discrete Gauss law**: G_kernel = Q_G/D = pi/6 | `hqvm_gravity_analysis_2.py` S10; `hqvm_gravity_common.py` `verify_gauss_law_bridge` | Q_G x G_kernel = D |
| 184 | **Plaquette curvature spectrum**: 1024 x C(6,k) for popcount k=0...6 | `hqvm_gravity_analysis_8.py` `section_a_plaquette_census` | Exhaustive over 256^2 byte pairs |
| 185 | **Plaquette census reproduces D=24**: sum of popcounts / (2\|Omega\|) = 24 | `hqvm_gravity_analysis_8.py` `section_a_plaquette_census` | Closed-form calculation |
| 186 | **Refractive Depth as Regge action**: tau_G matches closed form to relative precision 3.7e-16 | `hqvm_gravity_analysis_8.py` `section_b_regge_sum`, `section_e_chain_verification` | Executable verification |
| 187 | **k_eff = 3 from Regge sum**: spatial dimension emerges from BCH closure | `hqvm_gravity_analysis_8.py` `section_d_spectral_bridge` | Numerical readout |
| 188 | **Z2 BCH selection rule**: only even-order corrections survive projection | `hqvm_gravity_analysis_8.py` `section_c_bch_decomposition` | Symbolic computation (Dynkin truncation) |
| 189 | **Antimatter gravitoelectric invariants even**: D=24 holds for matter and antimatter | `hqvm_gravity_analysis_6.py` Section A | Exhaustive over 4096 states |
| 190 | **Antimatter gravitomagnetic invariants odd**: H_spin(C(s)) = -H_spin(s) for 2816 non-equatorial states | `hqvm_gravity_analysis_6.py` `verify_h_spin_under_C` | Exhaustive computational verification |
| 191 | **Constant-product falsification**: alpha_0 zeta = rho^4/(pi*sqrt(3)) independent of m_a | `hqvm_gravity_analysis_3.py` Part F; `hqvm_gravity_common.py` `verify_alpha_zeta_product` | Algebraic cancellation |

**Part IV subtotal (Tier B gravity kernel): 10 features**

---

## Part V: Electroweak Mass-Coordinate Law (Tier B)

From [Analysis_Compact_Geometry.md](../references/Analysis_Compact_Geometry.md). Kernel algebra proved in `hqvm_compact_geom_kernel.py`; mass law and audits reported by `hqvm_compact_geom_report.py` (computes via `hqvm_compact_geom_core.py`).

| # | Feature | Experiment | Method |
|---|---------|------------|--------|
| 192 | **Carrier-trace polynomial** for top, Higgs, Z, W masses with 6 coefficient orders (Delta through Delta^5) | `hqvm_compact_geom_core.py`; report SS3 | Fixed discrete grammar; no continuous fitting |
| 193 | **Max tick error 6.15e-9** at fifth order across four channels | `hqvm_compact_geom_report.py` SS4 | Comparison with PDG |
| 194 | **W/Z ratio recovers Delta to 8.34e-10** | `hqvm_compact_geom_report.py` SS5 | W/Z split back-solve |
| 195 | **Leave-one-out prediction**: each of H/Z/W predicted from other two to ~10^-5 relative | `hqvm_compact_geom_report.py` SS5 | Cross-validation |
| 196 | **Null-model audit**: rank-1 assignment gap ~11,000x over rank-2 | `hqvm_compact_geom_report.py` SS5.0 | Exhaustive over 4096 flag assignments |
| 197 | **Coefficient admissibility**: structural audit, no continuous fitting | `hqvm_compact_geom_report.py` SS4.4 | Structural audit |
| 198 | **Trace-free conditions**: Sum p_i = 0, Sum q_i = 0 | `hqvm_compact_geom_core.py` | Algebraic |
| 199 | **Coupling parametrizations**: lambda_H, g, g_Z, g', e, alpha_EW Delta, y_t to ~10^-5 relative | `hqvm_compact_geom_report.py` SS5.4 | From mass law at tree level |
| 200 | **Lepton carrier layer**: tau, mu, e coordinates via M_shell; unique path (5,8,14) | `hqvm_compact_geom_report.py` SS7; `hqvm_compact_geom_core.py` | Exhaustion over 680 valid triples |
| 201 | **148/51 closure**: K4 depth-4 (128) + full-byte len-2 (16) + micro paths (4) = 148 | `hqvm_compact_geom_report.py` SS7.1 | Exact rational |
| 202 | **Archetype closure**: electron dyadic closes at -51/256 | `hqvm_compact_geom_report.py` SS7.2 | Exact rational |
| 203 | **D_flow^2 quark ladder**: exact squared spacing \|d_flow\| = 1...6 for 6 quarks | `hqvm_compact_geom_report.py` SS8 | Empirical |
| 204 | **UV-IR conjugacy**: E_UV x E_IR = E_CS x v/(4pi^2) at all 4 stages | `hqvm_compact_geom_kernel.py` `run_kernel_verification` | Product = K to 9+ digits |
| 205 | **SU(3) sextet bracket closes** in 32-bit lifted space | `hqvm_compact_geom_report.py` SS6 | Phase-symmetrized check |

**Part V subtotal (Tier B electroweak): 14 features**

---

## Part VI: Continuous Gravity Phenomenology (Tier B)

Continuous field-theory predictions anchored on kernel invariants. Documented in [Analysis_Gravity.md](../references/Analysis_Gravity.md) and numerically closed in science repo scripts `hqvm_gravity_analysis_4.py` through `_10.py` and `hqvm_corrections_analysis_1.py`.

| # | Feature | Experiment | Method |
|---|---------|------------|--------|
| 206 | **Q_G = 4pi as quantum of gravity** (horizon normalization) | `hqvm_gravity_analysis_2.py` S12; `hqvm_gravity_common.py` | GNS + kernel ratio |
| 207 | **Virial condition 2T+V=0** as structural consequence of ancestry preservation | `hqvm_gravity_analysis_6.py` Section C | Kernel invariant D=24 |
| 208 | **Transport-corrected alpha matches CODATA to 0.043 ppb** | `hqvm_corrections_analysis_1.py` | Three geometric corrections in powers of Delta |
| 209 | **Delta self-consistency**: 3-factor reconstruction converges; D^3 fixed-point residual <10^-15 | `hqvm_compact_geom_core.py` | Iterative computation |
| 210 | **Position-dependent coupling**: G(psi) = G0 exp(g1 psi) with g1 = -0.6456 | `hqvm_gravity_analysis_4.py`; `hqvm_gravity_analysis_1.py` Part E | Three independent routes |
| 211 | **Weak-field G matches CODATA to 0.074 ppm** | `hqvm_gravity_analysis_1.py` Part E | G_pred = G_kernel exp(-tau_G)/v^2 |
| 212 | **c4 = -7/4** fixed by two independent kernel routes | `hqvm_gravity_analysis_1.py` Part A; `hqvm_gravity_analysis_3.py` Part E | STF + closure charge |
| 213 | **Per-family Refractive Depth uniformity**: zero variance across all 4 families | `hqvm_gravity_analysis_1.py` Part C | Verified |
| 214 | **Exact point-mass solution**: psi(s) = -(1/g1)ln(1-g1/s) | `hqvm_gravity_analysis_4.py` | Analytical + numerical endpoints |
| 215 | **Effective metric**: f = 1-2psi; Einstein tensor verified to 4.4e-16 | `hqvm_gravity_analysis_5.py` `section_full_einstein_tensor` | Numerical |
| 216 | **Modified Gauss law conservation** at all radii to 2.83e-16 | `hqvm_gravity_analysis_5.py` `verify_modified_gauss_law` | Numerical |
| 217 | **Self-energy theorem**: E_self = -Mc^2/4 (exact, finite) | `hqvm_gravity_analysis_6.py` Section C; `hqvm_gravity_analysis_7.py` | Exterior ODE |
| 218 | **Mass dressing**: M_obs = (4/5)M_bare (20% bound into field) | `hqvm_gravity_analysis_6.py` Section C | Self-consistent |
| 219 | **Chiral correction magnitude**: (4/75)psi^2 from constant anisotropy ratio | `hqvm_gravity_analysis_3.py` Part B; `hqvm_gravity_analysis_8.py` | Kernel invariant 2/75 |
| 220 | **PPN: gamma = 1** exactly (consistent with Cassini) | `hqvm_gravity_analysis_5.py` `section_ppn_analytical_final` | Leading deflection |
| 221 | **Nordtvedt parameter eta_N = 0** | `hqvm_gravity_analysis_5.py` `section_strong_equivalence` | G(psi) position-only dependent |
| 222 | **Mercury precession**: CGM/GR = 0.9999999973 (0.003 ppm) | `hqvm_gravity_analysis_5.py` `section_ppn_analytical_final` | Full metric geodesic |
| 223 | **Black hole shadow**: CGM predicts 80% of GR Schwarzschild area | `hqvm_gravity_analysis_4.py`; `hqvm_gravity_analysis_10.py` `section_e_holographic_bh` | Null geodesic computation |
| 224 | **Horizon at s_h ~ 1.695 r_g** (15.3% inward of Schwarzschild) | `hqvm_gravity_analysis_4.py`; `hqvm_gravity_analysis_7.py` | psi=1/2 condition |
| 225 | **Photon sphere at s_ph ~ 2.586 r_g** (vs 3.0 in GR) | `hqvm_gravity_analysis_4.py` `find_photon_sphere_spin` | Null geodesic |
| 226 | **Gravitational radiation**: quadrupole dominant; exactly 2 tensor polarization modes | `hqvm_gravity_analysis_6.py` Section B | Fourier decomposition |
| 227 | **Gravitational wave phase correction**: ~ -6.5% at v/c ~ 0.4 (GW150914) | `hqvm_gravity_analysis_5.py` `section_gw_strain_calibration` | Leading post-Newtonian |
| 228 | **Ringdown frequency shift**: fundamental ~ 12.5% above GR | `hqvm_gravity_analysis_6.py` Section B | Regge-Wheeler potential |
| 229 | **Vacuum impedance matching**: R+T=1 across sharp metric steps | `hqvm_gravity_analysis_7.py` Section B | Numerical integration |
| 230 | **UV-IR interface density depletes by ~10^-6 near horizon** | `hqvm_gravity_analysis_5.py`; `hqvm_gravity_analysis_10.py` | From E_ref formula |
| 231 | **Inflationary observables**: n_s ~ 0.972, r ~ 2.4e-3 in R^2 limit | `hqvm_gravity_analysis_9.py` | Slow-roll computation |
| 232 | **Asymptotic freedom of gravity**: d ln alpha_G / d ln mu ~ -0.017 < 0 | `hqvm_gravity_analysis_9.py` | Refractive Depth law |
| 233 | **Neutron star TOV with G(psi)**: R~15.4 km, M~1.25 Msun for gamma=2 polytrope | `hqvm_gravity_analysis_5.py` (TOV integration) | Numerical integration |
| 234 | **Redshift prediction for NS surface**: z_CGM ~ 0.200 vs z_GR ~ 0.235 | `hqvm_gravity_analysis_10.py` `section_a_redshift_channels` | Direct from metric |
| 240 | **Four-phase causal cycle**: Measure (CS), Vary (UNA), Retrieve (ONA), Commit (BU) | `hqvm_gravity_analysis_7.py` Section 6 | Byte transition decomposition |
| 242 | **E^2/5 efficiency**: rest-frame energy = M_obs c^2/4 = (1/5) M_bare c^2 | `hqvm_gravity_analysis_6.py` Section C | From self-energy theorem |
| 243 | **Intrinsic gravitational clock**: T_Z2 = (6/pi)GM/c^3 x surface gravity; vanishes at psi=1/2 | `hqvm_gravity_analysis_7.py` Section 4 | D=24 tied to speed of light |

**Part VI subtotal (Tier B continuous gravity): 32 features**

---

## Part VII: CGM Modal Structure (Tier C)

Formal modal-logic and manuscript-only results from [Analysis_Gravity.md](../references/Analysis_Gravity.md) and companion proof [15]. No dedicated experiment script in the science repo.

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 235 | **Five foundational conditions** logically independent in core modal system | Gravity Analysis App. A | Counterexample frames |
| 236 | **Consistency verified** via three-world Kripke frame | Gravity Analysis App. A | Model construction |
| 237 | **3D necessity**: n=3 unique dimension satisfying all 5 conditions | Gravity Analysis SS2.4; [15] | Formal proof |
| 238 | **SE(3) = SU(2) rtimes R^3** forced by bi-gyrogroup consistency from ONA | Gravity Analysis SS2.4; [15] | Formal proof |
| 239 | **sl(2) from BCH**: depth-4 commutator forces Lie algebra to close on 3 generators | Gravity Analysis SS2.4; [15] | Hall word exclusion |
| 241 | **Intelligence = BU closure** (preserve ancestry + identity + individuality) | Gravity Analysis SS2.1 | Operational definition |

**Part VII subtotal (Tier C formal): 6 features**

---

## Summary Statistics

### By verification tier

| Tier | Label | Count | What it means |
|------|-------|------:|---------------|
| **A** | Kernel pytest (this repo) | **165** | `tests/physics/`, `tests/test_hQVM_*.py`, Moments, Holography |
| **B** | Science executable | **72** | 16 wavefunction + 10 gravity kernel + 14 electroweak + 32 continuous gravity |
| **C** | Formal manuscript | **6** | Modal logic independence, 3D necessity, SE(3); proofs in [15] |
| | **Grand total** | **243** | |

### By domain

| Domain | Tier A | Tier B | Tier C | Total |
|--------|-------:|-------:|-------:|------:|
| Algebraic quantum computation | 130 | 16 | 0 | 146 |
| Discrete kernel dynamics | 27 | 10 | 0 | 37 |
| Hardware / native execution | 8 | 0 | 0 | 8 |
| Electroweak mass law | 0 | 14 | 0 | 14 |
| Continuous gravity phenomenology | 0 | 32 | 0 | 32 |
| CGM modal formal structure | 0 | 0 | 6 | 6 |
| **Total** | **165** | **72** | **6** | **243** |

### Cross-repo verification map

| Manuscript | Tier A overlap | Tier B science scripts | Tier C only |
|------------|----------------|------------------------|-------------|
| Analysis_hQVM_Wavefunction | K4 gates (Part I) partially overlap | `hqvm_wavefunction_1.py`, `_2.py` (T1-T10 exhaustive) | none |
| Analysis_Gravity | Physics/hQVM tests cover many kernel items | `hqvm_gravity_analysis_1-10.py`, `hqvm_gravity_common.py`, `hqvm_corrections_analysis_1.py` | App. A modal proofs (#235-239) |
| Analysis_Compact_Geometry | Shell structure in hQVM SDK tests | `hqvm_compact_geom_kernel.py`, `_core.py`, `_report.py` | none |

Run all gravity scripts and capture output:

```text
python experiments/hqvm_gravity_runner.py
# writes experiments/hqvm_gravity_analysis.txt
```

### What each tier definitively proves

**Tier A (165 features).** Kernel pytest establishes the Gyroscopic hQVM Kernel as an exactly solvable finite algebraic quantum system on standard silicon: 4096-state manifold, K4 gates, Hilbert lift to 10^-12, computational advantages, non-Clifford delta(BU), self-dual code, CGM constants bridge.

**Tier B (72 features).** Science repo executables extend verification to the three analysis manuscripts: wavefunction theorems T1-T10 on all Omega states, gravity kernel invariants (D=24, plaquette/Regge/tau_G), electroweak mass law to 6e-9 tick error, and continuous gravity phenomenology (PPN, shadow, TOV, GW) to numerical closure.

**Tier C (6 features).** Formal modal-logic results (independence of CS/UNA/ONA/BU, 3D necessity, SE(3) emergence) proved in manuscript [15], not wrapped as experiment scripts.

---

*Last updated from superintelligence verification reports, analysis manuscripts, and science repo experiment script headers. This inventory does not re-run any suite.*
