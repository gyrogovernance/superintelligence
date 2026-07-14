# Analysis: hQVM Cohomology

## Étale Cohomology and Grothendieck Structure in CGM and hQVM

## Abstract

The Holonomic Quantum Virtual Machine (hQVM) realizes the Common Governance Model (CGM) as a finite state carrier over a characteristic-2 transport base. The CGM construction is organized as a chain of formal layers, each isolating one facet of the requirement that ancestry stay preservable under recursive operations. This document is the cohomology layer of that chain: the level at which transport obstructions receive an algebraic classification beyond what graph connectivity records. Standard graph reachability reports only whether states connect. This document adds the algebraic classification of how generator restrictions obstruct connectivity and of why the Hilbert-space description outperforms the Boolean one. This document builds a finite covering system on the hQVM transition category to classify those obstructions and to compare the Boolean and Hilbert descriptions as two section classes on one site. The shell populations satisfy a discrete Poincaré duality, and comparing Boolean sections to Hilbert-lift sections on horizon ensembles yields the Grothendieck constant of order 2, K_G^R(2) = √2. The residual aperture Δ = 1 - ρ is the obstruction this covering detects, the residual non-closure parameter linking the BU monodromy to the aperture scale. The CHSH measurement on the bipartite carrier turns the constant-level Grothendieck correspondence into a directly falsifiable result.

---

## 1. Purpose and Scope

Étale cohomology was introduced to extract cohomological invariants from algebraic objects whose ordinary topology is too coarse for constant finite coefficients. The hQVM presents the same structural need in finite computational form. The carrier Ω is a finite state set, so ordinary point-set topology has limited explanatory content. The transition system contains richer local structure because each byte acts as a reversible local transport map, and the byte alphabet itself has a nontrivial fiber structure over the chirality register.

The CGM construction is realized in several formal layers: modal logic, Hilbert space, Lie and gyrogroup algebra, the finite hQVM machine, generator-restricted percolation, gravity and constants. Each layer isolates one facet of the ancestral-preservation requirement. The cohomology layer developed here supplies the classification of transport defects, a facet the other layers record only as adjacency or as a closure invariant. Its role is to state what kind of algebraic obstruction prevents a generator restriction from preserving ancestry globally, and to compare the Boolean and Hilbert descriptions of the carrier as two section classes on one site. The Hilbert lift plays the same role as a semidefinite-programming relaxation of a Boolean cut optimization: it enlarges the feasible section class while remaining computable on classical hardware.

The structural problem addressed here and in Grothendieck's program is the same: ordinary local data is too coarse to expose the global invariants that govern connectivity and closure, so one must pass to covers, classify the cocycles and obstructions that prevent global sections, establish a duality, evaluate traces, and read off a zeta function. Grothendieck solved this for algebraic varieties by replacing the Zariski topology with étale covers. The hQVM gives a finite constructive analogue of that categorical problem for a GF(2) state machine by replacing flat XOR transport with the family-fiber cover and the Hilbert lift. The shared constants (pi/2, square root of 2, the Krivine expression) are the scalar shadows of this shared pattern, the surface trace of the correspondence on the invariant values.

Standard graph reachability identifies whether states are connected. Grothendieck topology classifies how the connectivity is obstructed and specifies the algebraic structure of those obstructions. The hQVM transition graph records adjacency, but the parity obstruction and the Grothendieck CHSH gap require a cohomological classification of transport defects. This document defines a finite covering system on the hQVM transition category and studies observable sections on that site. The construction uses finite covers, finite fibers, group cohomology, and permutation trace formulas. It follows the organizing principle of étale cohomology while remaining a finite hQVM construction.

The analysis has three levels. The finite-site level defines covers and observable assignments on the hQVM transition category. The arithmetic-dynamical level evaluates fixed points, cycle zeta functions, and Lefschetz numbers of byte and word operators. The analytic level compares Boolean and Hilbert sections through the Grothendieck inequality and the CHSH test.

The construction uses finite characteristic-2 coefficients because the native chirality register is GF(2)^6. The closest classical étale analogue is the additive Artin-Schreier setting in characteristic p. The hQVM computation uses the finite transport category directly.

---

## 2. The hQVM Carrier as a Finite Cover

### 2.1 Carrier states

The hQVM state is a 24-bit carrier

```text
state24 = (A12 << 12) | B12
```

The two 12-bit components A12 and B12 are conjugate gyrophases. The reachable set Ω is the set of states generated from the rest state

```text
GENE_MAC_REST = 0xAAA555
```

under the byte transition rule. Exhaustive enumeration gives

```text
|Ω| = 4096
```

The carrier factorizes as

```text
Ω = U × V
|U| = 64
|V| = 64
|Ω| = 64² = 4096
```

The two factors U and V are the finite conjugate faces required by the CGM distinction between identity and individuality. Identity requires a recoverable record. Individuality requires operational displacement. The hQVM carrier realizes the two requirements by keeping an active face and a passive record face.

### 2.2 Chirality base

Each 12-bit face is arranged as six oriented dipole pairs. Comparing A12 and B12 pair by pair gives a 6-bit chirality word

```text
χ(s) ∈ GF(2)^6
```

The chirality map is

```text
χ: Ω → GF(2)^6
```

Each chirality value has 64 carrier states above it. The 64 values of GF(2)^6 form the base of the finite cover.

The 24-bit carrier contains redundant geometric information relative to the transport dynamics. The chirality base captures the transport dynamics without this redundant orientation data. The projection to GF(2)^6 is the topologically relevant quotient because the byte transition rule acts as XOR transport on this register while leaving the fiber structure invariant.

The shell of a state is the Hamming weight of its chirality word.

```text
shell(s) = popcount(χ(s))
```

Shells range from 0 to 6. Shell 0 is the equality horizon, where the two faces agree in every mode. Shell 6 is the complement horizon, where the two faces differ in every mode.

### 2.3 Byte transport

Each byte b defines a reversible transition T_b on Ω. The byte is first transcribed relative to the micro-archetype

```text
intron(b) = b XOR 0xAA
```

The six inner bits of the intron determine a 6-bit payload. The two boundary bits determine one of four family phases. The byte transition induces chirality transport

```text
χ(T_b(s)) = χ(s) XOR q6(b)
```

where q6(b) is the 6-bit transport value of b.

The map from bytes to transport values is 4-to-1.

```text
q6: Byte256 → GF(2)^6
|q6⁻¹(q)| = 4 for every q ∈ GF(2)^6
```

The four bytes over one transport value form a family fiber. The family group is the Klein four-group

```text
K4 = {id, S, C, F}
```

This gives a finite principal K4 cover of the transport base.

```text
Byte256 → GF(2)^6
fiber = K4
```

### 2.4 The hQVM transition site

The hQVM transition site is the category C with the following data.

Objects. All subsets X ⊆ Ω. The subsets used throughout this document are the following distinguished objects.

```text
Ω                    the full state space
Shell_k              for k = 0, ..., 6 (the seven shells)
H_eq = Shell_0       the equality horizon
H_comp = Shell_6     the complement horizon
Fiber_q = χ⁻¹(q)     for each q ∈ GF(2)^6 (the 64 chirality fibers)
```

Morphisms. Two types.

```text
1. Inclusions      i: A ↪ B whenever A ⊆ B (as subsets of Ω)
2. Byte restrictions T_b|_A : A → Ω for each byte b and object A,
   defined by T_b|_A(s) = T_b(s) for s ∈ A, provided T_b(A) ⊆ B
   for some object B (making it a morphism A → B in C)
```

Composition. Standard function composition. Inclusions compose as subset inclusions. Byte restrictions compose by applying successive transitions.

Pullbacks. For inclusions i: A ↪ B and j: C ↪ B, the pullback is A ∩ C with the evident inclusion maps. For a byte restriction T_b: A → B and an inclusion i: C ↪ B, the pullback is T_b⁻¹(C) ∩ A with the restriction of T_b to this set. Since objects are all subsets of Ω, every pullback is again an object, so the category is closed under fiber products.

Covering families. A finite family of inclusions {U_i ↪ X} covers X when

```text
⋃_i U_i = X
```

This is the standard coverage for a finite site. Byte restrictions enter the site through morphisms and pullbacks, separate from the covering structure. The coverage records that local observable assignments on a union of inclusions glue when they agree on overlaps.

---

## 3. Observable Sections on the hQVM Site

### 3.1 Section assignments

A local observable is defined on a subset of Ω reachable by a restricted generator set. A global observable is defined on all of Ω. The descent condition requires that local observable assignments agreeing on overlaps can combine into a global observable. The hQVM requires this formalism because the percolation analysis studies restricted generator sets, and tracking observable behavior under these restrictions necessitates local-to-global gluing rules.

A presheaf F assigns data F(X) to each object X in the hQVM site and assigns restriction maps along transition-compatible morphisms. F is a sheaf when compatible local data glue uniquely. The hQVM requirement of deterministic replayability fulfills the descent condition natively for state trajectories.

### 3.2 The transport span

For an allowed generator set A, the transport span is

```text
Q(A) = span_GF(2){q6(b): b ∈ A}
```

The rank of this subgroup is

```text
r(A) = dim_GF(2) Q(A)
```

The residual obstruction module is

```text
GF(2)^6 / Q(A)
```

Full transport coverage occurs when the quotient is zero.

### 3.3 The Boolean observable presheaf

Define F_bool: Cᵒᵖ → Set as follows.

On objects.

```text
F_bool(X) = {f: X → {±1} | f = W_m|_X for some nonzero mask m ∈ GF(2)^6}
```

Here W_m(s) = (-1)^(popcount(χ(s) ∧ m)) is the Walsh observable with mask m restricted to X.

On morphisms. For an inclusion i: A ↪ B, the restriction ρ_i: F_bool(B) → F_bool(A) is literal restriction of functions, ρ_i(f) = f|_A. For a byte restriction T_b: A → B, the pullback is ρ_{T_b}(f) = f ∘ T_b|_A.

This is a presheaf by construction. It is a sheaf for the site topology because Walsh observables are determined pointwise. If two sections agree on a cover, they agree globally.

The space of global sections F_bool(Ω) has dimension 63 (the nonconstant Walsh masks). On a chirality fiber Fiber_q, the chirality χ is constant, so every Walsh observable W_m is constant on Fiber_q. The image set {W_m|_{Fiber_q} : m ≠ 0} therefore has size at most 2, the two constant functions +1 and -1.

For an ensemble E ⊆ Ω, the face-face correlator built from these sections is

```text
C_E(m,n) = (1/|E|) Σ_{s ∈ E} W_m(A(s)) W_n(B(s))
```

These correlators are global sections of the Boolean correlation presheaf over E.

### 3.4 The Hilbert observable presheaf

The hQVM has a canonical Hilbert lift induced by the self-dual [12,6,2] binary code geometry of the mask space. Within the CGM three-dimensional, six-degree-of-freedom framework this code is an algebraic structure; it does not introduce an extra spatial dimension. In this lift, canonical word operators act as unitary operators on a complex vector space indexed by Ω. Define F_hilb: Cᵒᵖ → Set as follows.

On objects. F_hilb(X) is the set of operators on the subspace H_X := span{|s⟩ : s ∈ X} ⊆ ℂ^4096 generated by restrictions of canonical word unitaries U_W to H_X, closed under adjoint.

On morphisms. For an inclusion i: A ↪ B, the restriction is compression, ρ_i(O) = P_A O P_A, where P_A projects onto H_A. For byte restrictions, pull back by precomposition.

This is a presheaf. The descent condition that would make it a sheaf fails in general, because compressed operators from overlapping domains do not combine into a single global operator. That gluing failure is the structural counterpart of the Grothendieck gap: it is the same defect (local sections that refuse to glue into a global one) measured two ways, by gluing failure here and by the sign-versus-vector norm ratio in Section 3.5.

### 3.5 Section classes and the Grothendieck comparison

For any object X, define two correlators.

```text
C_bool(X)   max correlation matrix achievable by F_bool(X) sections
C_hilb(X)   max correlation matrix achievable by F_hilb(X) sections
```

The Grothendieck ratio K_G(X) = C_hilb(X) / C_bool(X) measures the gap between the two section classes on X. This is a computed comparison of two classes defined on the same site, obtained directly from the section correlators.

```text
X = H_eq or H_comp:   K_G = √2 (verified computationally)
X = Ω:                K_G undefined (C_bool = 0 by depth-2 typicality)
```

Boolean sections evaluate sign correlations. Hilbert sections evaluate inner-product correlations. The Grothendieck inequality compares these two section classes for the same coefficient matrix.

---

## 4. Cohomology and Obstruction in the Transport Cover

### 4.1 Group cohomology of the family cover

The family cover has deck group K4. With trivial action on the coefficient module GF(2)^6, the first group cohomology is

```text
H¹(K4, GF(2)^6) = Hom(K4, GF(2)^6)
```

K4 is isomorphic to GF(2)^2. Therefore

```text
dim_GF(2) H¹(K4, GF(2)^6)
=
dim_GF(2) Hom(GF(2)^2, GF(2)^6)
=
2 × 6
=
12
```

This computation runs on the family fiber. It evaluates group cohomology of the deck group, which is distinct from a Čech cohomology computation of a cover of Ω.

The 4-to-1 family cover generates a 12-dimensional space of cross-homomorphisms from the family fiber to the transport base. The bipartite carrier provides 12 degrees of freedom per face. The complementarity identity h + ab = 12 confirms that the carrier geometry saturates the dimension of the fiber-to-base cross-homomorphisms. Both quantities use the same chirality dimension d = 6 and the rank 2 of K4. This document records this dimensional matching as an observed coincidence of the two constructions.

### 4.1.1 The chirality transport as Artin-Schreier extension

The chirality transport rule χ(T_b(s)) = χ(s) ⊕ q6(b) is an additive action of GF(2)^6 on itself by translation. This is the characteristic-2 additive analogue of a Kummer extension. Where Kummer theory uses the multiplicative group and roots of unity, the additive theory uses the additive group and Artin-Schreier polynomials X² - X - α.

The orbit structure of the transport group (GF(2)^6, ⊕) acting on the base GF(2)^6 is as follows.

```text
The stabilizer of any q ∈ GF(2)^6 is trivial (translation has no
  fixed points except the identity)
The orbit of q under a rank-r subspace Q(A) has size 2^r
The fixed field of a subgroup Q(A) is the coset space
  GF(2)^6 / Q(A) of dimension 6 - r(A)
```

The parity functional parity: GF(2)^6 → GF(2), parity(q) = popcount(q) mod 2, is a nonzero homomorphism and therefore defines a class in H¹(GF(2)^6, GF(2)) ≅ Hom(GF(2)^6, GF(2)).

The even-weight subspace ker(parity) has rank 5. The quotient GF(2)^6 / ker(parity) ≅ GF(2) has dimension 1. This single bit of obstruction controls whether odd shells are reachable from shell 6.

This is the Artin-Schreier additive pattern in the hQVM. The chirality register GF(2)^6 is the additive base, the transport values q6(b) generate subspaces, and the parity homomorphism is the cohomological obstruction that partitions the state space into even-shell and odd-shell components. The obstruction is a linear homomorphism parity: GF(2)^6 → GF(2), whose kernel ker(parity) is a rank-5 subspace; the quotient GF(2)^6 / ker(parity) ≅ GF(2) is the single bit that controls odd-shell reachability.

### 4.2 Global sections and typicality

For the Boolean observable section class, global sections contain Walsh assignments that remain consistent across the selected cover. Under the uniform Ω ensemble, depth-two typicality makes all nonconstant face-face Walsh correlators vanish.

The verified depth-two future-cone entropy S_n from state s is

```text
S₀(s) = 0 bits
S₁(s) = 7 bits
S₂(s) = 12 bits
```

At depth two, every one of the 4096 states appears with equal multiplicity. This uniformization gives the zero Boolean CHSH value on Ω. Under the uniform measure on Ω, every nonconstant Walsh face observable has mean zero and every cross-face correlator C(m,n) equals 0. Therefore every Boolean CHSH expression built from these correlators equals 0.

### 4.3 Parity obstruction as a 1-cocycle

For a restricted byte set A, the transport span Q(A) controls reachable chirality values. The residual transport obstruction is

```text
Ob(A) = GF(2)^6 / Q(A)
```

The dimension of this quotient is

```text
dim Ob(A) = 6 - r(A)
```

When Ob(A) has positive dimension, some chirality directions are absent from the cover.

The parity functional is a homomorphism from the transport group to GF(2).

```text
parity(q) = popcount(q) mod 2
```

The even-weight transport values form the kernel of this homomorphism, a rank-5 subspace of GF(2)^6. The even-weight restriction restricts the dynamics to the kernel of this cocycle. The unreachable odd shells represent the non-trivial cohomology class.

Starting from shell 6, even-weight transport reaches only even shells. The resulting cluster size is

```text
|Reach| = 32² = 1024
```

This integrates the algebraic rank defect directly into the covering geometry.

---

## 5. Shell Exterior Grading and Duality

### 5.1 Shell census polynomial

The shell profile is determined by the six independent chirality modes. A shell-k state has k disagreeing modes. The number of chirality words of weight k is C(6,k), and each chirality word has 64 carrier states above it.

```text
|Shell k| = 64 × C(6,k)
```

The shell populations are

```text
64, 384, 960, 1280, 960, 384, 64
```

Dividing by 64 gives the finite shell census polynomial profile

```text
1, 6, 15, 20, 15, 6, 1
```

This is the coefficient vector of

```text
P(t) = (1 + t)^6
```

### 5.2 Exterior algebra grading

The six chirality modes define a finite exterior grading on the vector space V = GF(2)^6. Degree k corresponds to choosing k active disagreement modes from six. The shell census polynomial records the graded dimensions of the exterior algebra Λ•(V).

### 5.3 Poincaré duality via the wedge pairing

Let V = GF(2)^6 with standard basis e₁, ..., e₆. The exterior algebra Λ•(V) has graded components Λ^k(V) of dimension C(6,k), with Λ⁰(V) ≅ GF(2) and Λ⁶(V) ≅ GF(2).

Top form. Define the top form ω.

```text
ω = e₁ ∧ e₂ ∧ e₃ ∧ e₄ ∧ e₅ ∧ e₆ ∈ Λ⁶(V)
```

This is a basis for the one-dimensional top cohomology.

Pairing. For α ∈ Λ^k(V) and β ∈ Λ^(6-k)(V), define the pairing

```text
⟨α, β⟩ = (α ∧ β) / ω ∈ GF(2)
```

where division by ω means the following. The product α ∧ β is a scalar multiple of ω, and one extracts that scalar.

Non-degeneracy. If ⟨α, β⟩ = 0 for all β ∈ Λ^(6-k)(V), then α = 0. This follows from the standard property of the exterior algebra: the wedge product induces a perfect pairing Λ^k(V) × Λ^(6-k)(V) → Λ⁶(V) ≅ GF(2).

Consequence. dim Λ^k(V) = dim Λ^(6-k)(V), which forces |Shell_k| = |Shell_(6-k)| since each shell has 64 states per chirality word. The wedge pairing determines the shell symmetry. The verified shell populations satisfy the same relation.

```text
|Shell k| = |Shell 6-k|
```

This duality is the discrete Poincaré duality of the exterior algebra Λ•(V) on V = GF(2)^6. It is the cohomology of the six-dimensional torus (the classifying space of the chirality group), inherited by the hQVM chirality base, and fixed by the self-dual [12,6,2] code structure that generates the register. The Hodge star isomorphism Λ^k ≅ Λ^(6-k) supplies the pairing.

### 5.4 Euler characteristic

The Euler characteristic is

```text
χ = Σ_{k=0}^6 (-1)^k dim Λ^k(V) = Σ_{k=0}^6 (-1)^k C(6,k) = (1 - 1)^6 = 0
```

This vanishes identically, independent of any representation choice. It means the shell-graded carrier has equal total even and odd dimension, consistent with the balance between horizon shells (k = 0, k = 6) and bulk shells (k = 1, ..., 5). This matches the computational output

```text
P(-1) = 0
```

The vanishing Euler characteristic is the finite shell expression of balanced egress and ingress in the hQVM carrier.

### 5.5 Complementarity invariant

Every carrier state s carries two integer distances built from the same pairwise chirality data: horizon_distance(s), the Hamming distance from one constitutional pole of the carrier, and ab_distance(s), the Hamming distance between the two faces A12 and B12. They satisfy

```text
horizon_distance(s) + ab_distance(s) = 12
```

The value 12 equals two contributions for each of the six chirality modes. The invariant records that distance from one constitutional pole and distance between the two faces are complementary readings of the same pairwise chirality data.

---

## 6. Lefschetz Trace and Byte Dynamics

### 6.0 The shell grading as a finite Lefschetz grading

Define the shell-graded real vector space

```text
C^k = ℝ^{Shell_k}   real-valued functions on shell k
dim C^k = C(6,k) × 64
```

For any byte operator T_b, a permutation of Ω, T_b acts on ℝ^Ω by permutation of basis vectors. It maps shell k to shell k' in general and therefore leaves the grading without a chain map. The graded Lefschetz number L(T_b) defined below is therefore a shell-parity weighted fixed-point count: the alternating sum of fixed points across shells, a Lefschetz-type grading on Fix(T_b) expressed as a trace on the full permutation representation.

### 6.1 Finite permutation trace

Every byte transition is a permutation of Ω. For a finite permutation T acting on the carrier basis, the trace of the permutation matrix equals the number of fixed carrier states.

```text
Tr(T on carrier basis) = |Fix(T)|
```

A graded Lefschetz number uses shell parity.

```text
L(T) = Σ_k (-1)^k |Fix(T) ∩ Shell k|
```

This finite trace formula counts graded fixed points of permutations.

### 6.2 Byte fixed-point census

Exhaustive enumeration over all 256 bytes gives two classes.

```text
252 bytes have no fixed points.

4 bytes have 64 fixed points.
```

The four fixed-point bytes are

```text
0x2B
0x54
0xAA
0xD5
```

They are precisely the zero-transport bytes.

```text
q6(b) = 0
```

Their fixed sets lie on the two horizons.

```text
0xAA and 0x54 fix shell 0.

0x2B and 0xD5 fix shell 6.
```

No bulk shell contains fixed points for these bytes.

### 6.3 Lefschetz values of byte operators

The four stabilizer bytes fix the horizon states. The horizons are the states invariant under the zero-transport operation. Generic bytes have no fixed points because they move every state off the invariant set. The Lefschetz trace separates the invariant states from the transit states.

The Lefschetz values are

```text
L(T_b) = 64 for the four zero-transport stabilizers.

L(T_b) = 0 for the remaining 252 bytes.
```

The mean value over the full alphabet is

```text
(4 × 64) / 256 = 1
```

The fixed-point count and Lefschetz value have correlation 1 because all fixed points occur in one shell for each stabilizer.

### 6.4 Byte cycle zeta functions

A generic byte has cycle signature

```text
{4: 1024}
```

This means the byte decomposes Ω into 1024 four-cycles. Its fixed-point counts are

```text
Fix(T^n) = 4096 when 4 divides n
Fix(T^n) = 0 otherwise
```

The dynamical zeta function is

```text
Z_T(t) = (1 - t^4)^(-1024)
```

For a fixed-point-free depth-four word W2, W2', or F, the cycle signature is

```text
{2: 2048}
```

The corresponding zeta function is

```text
Z(t) = (1 - t^2)^(-2048)
```

All byte and word eigenvalues are roots of unity because all operators are finite permutations.

Connection to Weil-style zeta. In the Weil conjectures, the zeta function of a variety over F_q is controlled by fixed points of the Frobenius morphism F: x ↦ x^q. Here the iterates T^n play the role of F^n, and Fix(T^n) plays the role of N_n (point counts over F_{q^n}). The structural parallel is as follows.

| Weil zeta | hQVM dynamical zeta |
|-----------|---------------------|
| Variety X over F_q | Carrier Ω |
| Frobenius F | Byte/word operator T |
| N_n = \|X(F_{q^n})\| | Fix(T^n) |
| ζ(X,s) = exp(Σ N_n/n · q^{-ns}) | Z_T(t) = exp(Σ Fix(T^n)/n · t^n) |
| Riemann hypothesis: \|α\| = q^{i/2} | All eigenvalues on unit circle |

The hQVM zeta satisfies the exact analogue of the rationality property, being a finite product of terms (1 - t^m)^(-c), and the functional equation Z(1/t) = t^(-N) Z(t). The Riemann hypothesis analogue, that all eigenvalues lie on the unit circle, holds trivially for finite permutations.

### 6.5 Depth-four closure

For every micro-reference m, the three canonical closure operators W2(m), W2'(m), and F(m) are fixed-point-free involutions.

```text
W2(m)^2 = id
W2'(m)^2 = id
F(m)^2 = id
```

They have the same cycle signature.

```text
{2: 2048}
```

Their shell actions differ.

```text
W2(m) maps shell k to shell 6-k.

W2'(m) maps shell k to shell 6-k.

F(m) preserves shell k.
```

The fixed-point trace sees all three as Lefschetz-zero involutions. Their distinction is carried by the pairing map between states.

---

## 7. Percolation as Cohomological Descent

### 7.1 Generator restriction

A generator-restricted hQVM uses a subset A of the 256-byte alphabet. The transport span over A is

```text
Q(A) = span_GF(2){q6(b): b ∈ A}
```

The rank r(A) determines the chirality directions covered by A.

### 7.2 Square-root reachability

Descent theory asks whether local data over a cover can reconstruct the global object. The local data are the chirality directions spanned by the allowed generators. Descent fails when local transport directions do not span the full base.

Under fiber-complete restriction, the reachable set factors through the two conjugate faces.

```text
Reach(A) = U_A × V_A
|U_A| = |V_A| = 2^r(A)
```

For r(A) at least 1,

```text
|Reach(A)| = (2^r(A))²
```

Full reachability occurs when r(A) = 6.

```text
|Reach(A)| = 64² = 4096
```

The square-root cluster size measures the fraction of the global object recovered by the partial descent.

The product form survives fiber-incomplete selection. An alphabet with one family byte per transport value, spanning transport rank 2, reaches 16 states as a 4 by 4 product.

```text
n_allow = 4   q-span rank = 2   fiber_complete = False
Reach = 16 = 4 × 4   per-factor rank = log2(16)/2 = 2 = r(q6)
```

The per-factor rank equals the transport-span rank r(q6) whether or not the family fibers are complete. Fiber-incompleteness does not inflate the per-factor rank, and the identity |Reach| = (2^r)² holds with r = r(q6) in both cases.

### 7.3 Parity obstruction

The even-weight transport values form the kernel of the parity functional. The quotient has one dimension.

```text
GF(2)^6 / ker(parity) ≅ GF(2)
```

The residual quotient prevents odd-shell access. Starting from shell 6, even-weight transport reaches only even shells.

The parity obstruction is a specific descent failure where even-weight generators provide local sections that agree on some overlaps but cannot glue to reach odd shells. The resulting cluster size is

```text
|Reach| = 32² = 1024
```

### 7.4 Coverage hierarchy

The percolation hierarchy refines transport descent. The following events require increasing amounts of generator coverage.

1. Horizon spanning requires a path from shell 6 to shell 0.
2. Full reachability requires r(A) = 6 with shell-parity access.
3. Defect-spectrum completion requires all seven transport defect weights.
4. Two-step uniformization requires all states to appear under length-two words.
5. Holonomy transport requires availability of depth-four closure words.

The verified 50 percent onset estimates under independent byte inclusion are

```text
E_span          p ≈ 0.022
E_full          p ≈ 0.029
E_spectrum      p ≈ 0.054
two-step cover  p ≈ 0.7
word closure    governed by 1 - (1 - p^4)^64
```

The aperture gap Δ is close to the weak spanning threshold.

```text
Δ = 0.0206995539
p_c(span) / Δ ≈ 1.04
```

This comparison relates the continuum aperture scale to the finite generator-spanning onset. A cohomological derivation of Δ requires a norm map from phase holonomy to finite cochain defect.

---

## 8. Boolean and Hilbert Sections

### 8.1 CHSH coefficient matrix

The CHSH test uses the coefficient matrix

```text
M = [[1, 1],
     [1,-1]]
```

For sign-valued observables, the Boolean value is bounded by 2.

```text
CHSH_Bool ≤ 2
```

For Hilbert-space observables with spectra ±1, the Tsirelson bound is

```text
CHSH_Hilbert ≤ 2√2
```

### 8.2 Boolean values on hQVM ensembles

The Boolean CHSH value depends on the ensemble E over which Walsh correlators are averaged.

The verified values are

```text
uniform Ω:          0
shell 0:            2
shell 1:            1.6667
shell 2:            2
shell 3:            1.2000
shell 4:            2
shell 5:            1.6667
shell 6:            2
fixed chirality:    2
```

The uniform value is zero because depth-two uniformization makes all nonconstant face-face Walsh correlators vanish over Ω.

Horizon and fixed-chirality ensembles have deterministic face relations. Their Walsh correlations reach the classical CHSH value 2.

The Hilbert optimum is constant at 2√2 across all ensembles, so the ensemble ratio Kg = CHSH_Hilbert / CHSH_Bool is a property of the Boolean ensemble alone. The measured ratios are as follows.

```text
ensemble           Boolean CHSH   Kg = 2√2 / Boolean
shell 0/2/4/6      2.0000         1.4142 = √2
shell 1/5          1.6667 = 5/3   1.6971
shell 3 (equator)  1.2000         2.3570
fixed chirality    2.0000         1.4142 = √2
```

The even shells and the fixed-chirality and horizon ensembles recover the pole value Kg = √2 = K_G^R(2). The odd shells 1 and 5 give a Boolean CHSH of 5/3, and the equatorial shell 3, which has the largest population (1280 states), gives the weakest Boolean CHSH at 1.2 and the largest ratio 2.357. The variation in Kg away from √2 reflects the Boolean ensemble weakening in the bulk while the Hilbert optimum stays fixed.

### 8.3 Hilbert value in the canonical lift

The canonical Hilbert lift gives the Tsirelson value on Bell-pair states and on graph-state pairs.

```text
CHSH_Hilbert = 2.828427124746
```

The residual from 2√2 is below 10⁻¹⁵ in the reported computations.

### 8.4 Grothendieck ratio of order 2

On horizon and fixed-chirality ensembles,

```text
CHSH_Hilbert / CHSH_Bool
=
2√2 / 2
=
√2
```

The value √2 equals the real Grothendieck constant of order 2.

```text
K_G^R(2) = √2
```

Order 2 is the unique finite dimension whose real Grothendieck constant is established in closed form (Krivine, 1979). Published bound tables list K_G^R(2) = √2 in both the lower and upper columns, while every other finite dimension leaves a strict gap between its bounds. The machine-precision equality measured here therefore instantiates the one Grothendieck value known in closed form. In the Tsirelson bound CHSH_Hilbert ≤ K_G^R(2d²) for local dimension d per party, the hQVM 2x2 CHSH uses one binary outcome per party (d = 1) and lands on K_G^R(2), the smallest nontrivial case and the only one with a constant established in closed form. Larger local dimension is a lead for future measurement, outside the scope of this computation.

One-step evolution from a fixed state has a 2-to-1 shadow. The 256 bytes produce 128 distinct next states. The unresolved bit is carried by the family phase of the applied byte in the ledger, a ledger field outside the state coordinates of Ω. The Boolean Walsh sections evaluate the CHSH matrix over the discrete GF(2)^6 topology. The Hilbert lift evaluates the same matrix over the continuous inner-product space generated by the self-dual code. On horizon ensembles, the Boolean optimization hits the classical bound of 2, while the Hilbert lift hits the Tsirelson bound of 2√2. The hQVM provides a finite physical realization of K_G^R(2) = √2, measuring the metric expansion required to lift the discrete transport graph into a unitary state space.

The gap vanishes on uniform Ω because depth-two uniformization destroys the correlation structure that the obstruction acts upon.

### 8.5 The Grothendieck relaxation on a classical carrier

The Boolean section class F_bool optimizes over sign observables. For a fixed coefficient matrix, this is an optimization over {±1}-valued assignments, the cut-norm or infinity-to-one type problem. The Hilbert section class F_hilb optimizes over inner products of unit vectors, equivalently over plus-minus Hermitian observables in the Hilbert representation. Replacing sign factorizations by Hilbert-space factorizations is the standard Grothendieck relaxation.

Grothendieck's inequality states that this relaxation enlarges the optimum by at most a universal constant factor, and the smallest such factor in dimension 2 is K_G^R(2) = square root of 2. The inequality was proved in functional analysis as a bound between tensor norms; its later role as a semidefinite-programming relaxation principle and as the bounding constant for quantum Bell inequality violation came from subsequent work, not from Grothendieck. The hQVM hosts two section classes on one finite carrier, and the Hilbert lift realizes correlations unattainable by Boolean Walsh sections.

On horizon ensembles, CHSH_Bool = 2 while CHSH_Hilb = 2 times square root of 2, so the strict ratio square root of 2 is achieved inside a finite, replayable state machine. This is a concrete instance of the Grothendieck relaxation gap on a classical carrier. The hQVM is named a virtual machine for this reason: the lift operates on a deterministic finite kernel, and the relaxation gap is structural and independent of hardware.

### 8.6 Hyperplane rounding

Grothendieck-type proofs use the sign of random hyperplane projections. For unit vectors x and y,

```text
E[sign(<g,x>) sign(<g,y>)]
=
(2/π) arcsin(<x,y>)
```

The hQVM chirality register gives 64 native sign vectors in R^6. Monte Carlo evaluation over these vectors gives

```text
fitted coefficient = 0.6367684
target 2/π         = 0.6366198
relative error     = 2.3 × 10^-4
```

The Walsh-Hadamard transform on the chirality register is orthogonal, and the finite sign ensemble follows the expected arcsin rounding relation within sampling error.

### 8.6 Localization of the gap to the CHSH projection

The sharp √2 gap is a property of the CHSH 2x2 projection, not of the full observable algebra. Comparing the Boolean cut norm and the Hilbert relaxation of the complete 63 by 63 mask-by-mask correlation matrix C on each conditioned ensemble gives a Grothendieck ratio of 1.

```text
ensemble             ||C||_F   bool_lb   hilb    ratio
uniform Ω            0.0000    0.0000    0.0000  undefined (typicality)
complement horizon   7.9373    63.0000   63.0000 1.0000
shell 0              7.9373    63.0000   63.0000 1.0000
shell 3              1.4832    7.0000    7.0000  1.0000
shell 6              7.9373    63.0000   63.0000 1.0000
```

On a single conditioned ensemble the Boolean and Hilbert optima of the full observable correlation matrix coincide, so the ambient observable algebra carries no Grothendieck gap. The gap of √2 appears only in the CHSH 2x2 projection of section 8.4. This localizes where the integrality gap lives in the section classes of section 3. The structural reason is that the CHSH matrix M isolates one Bell pair (one AB mode pair) and therefore one bipartition, so the Boolean sign optimum stays at 2 while the Hilbert inner-product optimum reaches 2√2. The full 63x63 matrix averages over all 63 mask correlators across all mode pairs; under that averaging the Boolean sign optimum and the Hilbert inner-product optimum coincide, and the gap collapses to 1. The integrality gap survives only when a single bipartition is singled out.

### 8.7 Depth closure and gauge triviality on the uniform ensemble

The Boolean gap is present only under conditioning. Breadth-first evolution from the rest state over the full 256-byte alphabet gives the depth-dependent Boolean CHSH profile.

```text
depth 0:   2.0000
depth 1:   0.0000
depth 2:   0.0000
depth 3:   0.0000
depth 4:   0.0000
```

The full-alphabet closure reaches the uniform bulk by depth 1, so all face correlators vanish and the Boolean CHSH is zero for every depth beyond the rest state. The integrality gap exists on conditioned horizon and fixed-chirality ensembles and closes under unconditioned depth evolution.

The K4 family gauge acts trivially on the uniform ensemble. Applying each gate to the full carrier gives image CHSH zero.

```text
gate id:  0.0000
gate S:   0.0000
gate C:   0.0000
gate F:   0.0000
```

Every K4 gate maps uniform Ω to uniform Ω, so the gauge leaves the uniform-ensemble CHSH invariant at zero.

### 8.8 A second finite Grothendieck instance on the horizons

The carrier carries a second finite Grothendieck object, distinct from the CHSH channel. It is the bipartite transition matrix from the complement horizon H_comp (shell 6) to the equality horizon H_eq (shell 0). Within a single horizon the transition rule is 1-regular, so the single-horizon graph has K(G) = 1 and its cut norm degenerates to total mass. The nontrivial instance is the 64 by 64 bipartite matrix between the two horizons.

```text
H_comp → H_eq total transitions:   256 (4 per state, uniform rows and columns)
unsigned cut norm:  256   Boolean opt 1024   SDP lower bound 128
signed cut norm:    184   Boolean opt  736   SDP lower bound 128
```

The signed matrix uses the L0-bit sign. Both SDP lower bounds collapse to 128 because the matrices are rank-deficient. The cut norm exceeds the SDP lower bound, so the Grothendieck ratio on this instance is below 1 and does not exceed the CHSH channel. This is a distinct finite Grothendieck instance on the same carrier, and it introduces no new constant. The CHSH 2x2 projection remains the place where the gap equals √2.

---

## 9. Krivine Constant and CGM Thresholds

### 9.1 UNA rapidity

The CGM Unity Non-Absolute threshold is

```text
u_p = 1/√2
```

The associated rapidity is

```text
artanh(u_p) = artanh(1/√2)
```

The verified identity is

```text
artanh(1/√2) = arsinh(1) = ln(1 + √2)
```

### 9.2 Krivine expression

The Krivine upper bound for the real Grothendieck constant is

```text
K_Krivine = π / (2 ln(1 + √2))
```

Using the rapidity identity gives

```text
K_Krivine = (π/2) / arsinh(1)
```

The verified identity ln(1 + √2) = arsinh(1) = artanh(1/√2) relates the Krivine denominator to the UNA threshold u_p = 1/√2 through a shared hyperbolic angle. This document records the identity among constants.

### 9.3 Gap spectrum

The hQVM and CGM constants appearing in this analysis have distinct roles.

```text
K_G^R(2) = √2
role: CHSH Hilbert-to-Boolean ratio on horizons

π/2
role: Common Source horizon threshold

K_Krivine = π / (2 ln(1 + √2))
role: general Grothendieck upper-bound expression

arsinh(1)
role: UNA rapidity scale

Δ = 0.0206995539
role: CGM aperture gap

δ_BU = 0.19534217658
role: BU dual-pole monodromy defect
```

These quantities enter one architecture through different measurements and normalizations.

---

## 10. Aperture and Holonomy

### 10.1 CGM aperture data

The CGM aperture scale is

```text
m_a = 1 / (2√(2π))
```

The horizon flux invariant is

```text
Q_G = 4π
```

They satisfy

```text
Q_G × m_a² = 1/2
```

The BU dual-pole monodromy defect is

```text
δ_BU = 0.19534217658
```

The closure ratio and aperture gap are

```text
ρ = δ_BU / m_a = 0.9793004461

Δ = 1 - ρ = 0.0206995539
```

### 10.2 Trace-angle recovery

The monodromy defect is recovered from the trace of the SU(2) half-loop.

```text
cos(δ_BU / 2) = (1/2) Re Tr(U_half)
```

The computation gives

```text
δ_BU from trace = 0.19534217658000036
absolute difference = 3.6 × 10^-16
```

The full loop has trace 2 and closes to the identity in the evaluated representation.

### 10.3 Aperture and finite cohomology

The finite cohomology supplies discrete obstruction measures such as

```text
dim GF(2)^6 / Q(A)

dim H¹(K4, GF(2)^6)

parity quotient GF(2)^6 / ker(parity)
```

The continuum aperture Δ supplies the residual phase opening of the BU holonomy. The bridge between them is the closure-ratio identity of Section 10.1,

```text
Δ = 1 - ρ = 1 - δ_BU / m_a
```

which expresses the aperture directly as the residual closure fraction of the BU monodromy against the aperture scale. The finite obstruction dim GF(2)^6 / Q(A) is the algebraic form of the same non-closure on the transport cover: it measures the rank defect that remains after the included generators act, just as Δ measures the phase defect that remains after the balanced loop. Both describe the gap between the partial restriction and full closure in their respective algebras. The finite and continuum quantities are therefore linked by the closure-ratio identity, not by a separately postulated map.

---

## 11. Physical Readout from the Same Geometry

### 11.1 Electromagnetic kernel coupling

The CGM electromagnetic kernel coupling is

```text
α₀ = δ_BU⁴ / m_a
```

The evaluated value is

```text
α₀ = 0.007299683322
```

The formula uses the BU monodromy defect and the aperture scale.

### 11.2 Gravitational aperture parameter

The gravitational aperture parameter is

```text
ζ = 8 / (m_a√3)
```

The evaluated value is

```text
ζ = 23.1552401459
```

The kernel-level product identity is

```text
α₀ × ζ = ρ⁴ / (π√3)
```

The computation verifies this identity to numerical precision.

### 11.3 Shell partition in the physical readout

The two horizon shells have zero bulk anisotropy in the hQVM gravitational readout. The five bulk shells carry the symmetric trace-free anisotropy used in the gravitational attenuation calculation.

The CHSH shell table reads the same shell geometry through correlation strength. Horizon and fixed-chirality ensembles reach the Boolean CHSH value 2. Bulk shells reduce the Boolean value, with the equatorial shell giving the smallest reported value.

The shared carrier geometry supplies both readings.

---

## 12. Computational Verification

The following computations are recorded in `experiments/hqvm_Cohomology_analysis_results.txt`.

### 12.1 Constant identities

The constant audit verifies

```text
Q_G × m_a² = 1/2

m_a² × 4π² = π/2

ln(1 + √2) = arsinh(1)

ln(1 + √2) = artanh(1/√2)

1 + √2 = tan(3π/8)

α₀ × ζ = ρ⁴ / (π√3)
```

All listed identities pass at tolerance 10^-12.

### 12.2 Cohomological and shell computations

The shell profile is

```text
1, 6, 15, 20, 15, 6, 1
```

The shell Euler characteristic is

```text
0
```

The complementarity invariant holds for all 4096 states.

```text
horizon_distance + ab_distance = 12
```

The group cohomology computation gives

```text
dim H¹(K4, GF(2)^6) = 12
```

### 12.3 Lefschetz and zeta computations

The byte fixed-point census gives

```text
252 pure order-four byte permutations

4 horizon stabilizer bytes
```

The generic byte zeta form is

```text
(1 - t^4)^(-1024)
```

The depth-four word zeta form is

```text
(1 - t^2)^(-2048)
```

### 12.4 Grothendieck and CHSH computations

The Hilbert CHSH value is

```text
2.828427124746
```

The horizon Boolean CHSH value is

```text
2.0000
```

The ratio is

```text
1.414213562373 = √2
```

The arcsin rounding coefficient on native chirality vectors is

```text
0.6367684
```

The target value is

```text
2/π = 0.6366198
```

---

## 13. Failure Conditions

The finite-site construction would fail if the hQVM transition category did not preserve the stated cover structure. A direct counterexample would be a transport value whose byte fiber has size other than 4, or a K4 family action that fails to act transitively on the fiber.

The shell cohomology profile would fail if exhaustive enumeration over Ω produced shell populations different from

```text
64, 384, 960, 1280, 960, 384, 64
```

The K4 cohomology computation would fail if the family action used a nontrivial module action on GF(2)^6 without updating the cocycle computation.

The transport obstruction analysis would fail if a fiber-complete restricted alphabet with rank r produced a reachable set whose size differs from

```text
(2^r)²
```

for r at least 1.

The Lefschetz trace computation would fail if any nonzero-transport byte had a fixed point, or if any zero-transport byte failed to fix one complete horizon.

The CHSH Grothendieck ratio would fail if the canonical Hilbert lift did not reach 2√2, or if the horizon Boolean Walsh correlators did not reach 2.

The aperture bridge would fail if the closure-ratio identity Δ = 1 - δ_BU / m_a did not hold for the stated constants, or if the finite transport obstruction dim GF(2)^6 / Q(A) ceased to describe the same non-closure that Δ measures.

---

## 14. Reproducibility

The computations use the scripts

```text
experiments/hqvm_Cohomology_analysis_1.py
experiments/hqvm_Cohomology_analysis_2.py
experiments/hqvm_Cohomology_analysis_3.py
experiments/hqvm_Cohomology_analysis_4.py
```

The combined output is

```text
experiments/hqvm_Cohomology_analysis_results.txt
```

The fixed random seed is

```text
20260702
```

Exhaustive enumerations run over all 4096 states of Ω and all 256 byte operators. Monte Carlo computations report their sample sizes and residuals in the output file.

---

## References

Grothendieck, A. (1953). Resume de la theorie metrique des produits tensoriels topologiques. Bol. Soc. Mat. Sao Paulo, 8, 1-79.

Grothendieck, A. (1960). The cohomology theory of abstract algebraic varieties. Proceedings of the International Congress of Mathematicians, Edinburgh 1958, 103-118.

Artin, M., Grothendieck, A., and Verdier, J. L. (1972). Theorie des topos et cohomologie etale des schemas. SGA 4. Lecture Notes in Mathematics 269, 270, and 305. Springer.

Deligne, P. (1974). La conjecture de Weil I. Publications Mathematiques de l'IHES, 43, 273-307.

Deligne, P. (1980). La conjecture de Weil II. Publications Mathematiques de l'IHES, 52, 137-252.

Milne, J. S. (1980). Etale Cohomology. Princeton University Press.

Krivine, J. L. (1979). Constantes de Grothendieck et fonctions de type positif sur les spheres. Advances in Mathematics, 31(1), 16-30.

Tsirelson, B. S. (1980). Quantum generalizations of Bell's inequality. Letters in Mathematical Physics, 4(2), 93-100.

Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. DOI: 10.5281/zenodo.17521384.