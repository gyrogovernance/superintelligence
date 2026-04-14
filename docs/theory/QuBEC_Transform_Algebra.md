# Native Transform Algebra of QuBEC Climate Dynamics
## Kernel Semantics, Spectral Structure, Arithmetic Charts, and Execution Consequences

---

## Scope

This document formalizes the transform algebra of the occupied QuBEC: the exact relation between the kernel transition law, the chirality spectral transform, the shell radial transform, the gauge character transform, and the arithmetic K4 contraction chart. It derives the execution consequences of that structure, including structured operator lowering, symbolic cost semantics, and interoperability with external tensors.

The document is organized in four parts. Part I establishes the semantic foundation and canonical coordinates. Part II develops the four native transforms and their unification. Part III derives execution consequences: quotient classes, algorithms, and symbolic cost. Part IV places the transform algebra inside the broader closure geometry of the CGM constants.

This document complements the QuBEC Climate Dynamics theory document, the aQPU Kernel SDK specification, and the Gyroscopic ASI Holographic Formalization. It does not repeat kernel physics established in those documents except where needed to ground the transform definitions.

---

## Notation

The following symbols are used throughout.

| Symbol | Meaning |
|---|---|
| Ω | Reachable manifold, \|Ω\| = 4096 |
| C₆₄ | Self-dual [12,6,2] mask code, 64 codewords |
| GF(2)⁶ | Six-dimensional binary field, chirality register space |
| χ | Chirality word, χ ∈ GF(2)⁶ |
| N | Shell index, N = popcount(χ) ∈ {0,…,6} |
| c | Boundary anchor, c ∈ GF(2)⁶ |
| q | Byte q-charge, q ∈ GF(2)⁶ |
| K4 | Klein four-group of intrinsic gates, K4 ≅ GF(2)² |
| η, ηᵢ | Isotropic and per-axis chirality damping parameters |
| ξ_A, ξ_B | Gauge damping parameters on the two K4 axes |
| m_a | Observational aperture, m_a = 1/(2√(2π)) |
| δ_BU | BU dual-pole monodromy |
| ρ | Closure ratio, ρ = δ_BU/m_a |
| Δ | Aperture gap, Δ = 1 − ρ |
| Q_G | Quantum gravity invariant, Q_G = 4π |
| B | Arithmetic radix, B = 2¹⁶ = 65536 |
| ⊕ | Bitwise XOR |

---

# Part I. Semantic Foundation

---

## 1. Reference Semantics

### 1.1 Normative surface

The exact semantics of the computational medium are defined by the kernel transition law and its exact observables. The reference implementation in `src/constants.py`, `src/api.py`, `src/kernel.py`, and `src/sdk.py` constitutes the normative semantic surface. All transforms formalized here are exact charts of the structure defined at that layer.

### 1.2 Byte semantics

Every byte b is transcribed into an intron:

```
intron = b ⊕ 0xAA
```

The intron decomposes into:

- family bits: positions 0 and 7, determining the K4 gauge phase
- payload bits: positions 1 through 6, determining the q-charge

The payload expands into a 12-bit mask by dipole-pair projection. Payload bit i maps to pair (2i, 2i+1) in the mask. The state update law is:

```
A_mut  = A12 ⊕ mask12
A_next = B12 ⊕ invert_a
B_next = A_mut ⊕ invert_b
```

where invert_a and invert_b are determined by the family bits.

### 1.3 Reachable manifold

The reachable manifold Ω consists of exactly 4096 states accessible from the rest state GENE_MAC_REST = 0xAAA555. It has product form:

```
Ω = U × V,  |U| = |V| = 64,  |Ω| = 64² = 4096
```

The two 64-state horizons are:

- complement horizon: A12 = B12 ⊕ 0xFFF (maximal chirality, N = 6)
- equality horizon: A12 = B12 (zero chirality, N = 0)

The complementarity invariant holds on all of Ω:

```
horizon_distance + ab_distance = 12
```

Every reachable component has constant density 0.5 (popcount = 6 out of 12 bits).

### 1.4 Canonical coordinates on Ω

The product form Ω = U × V admits a canonical decomposition of every state s into three coordinates.

**Boundary anchor** c ∈ GF(2)⁶: the coset representative, indexing which element of C₆₄ the U-component lies in.

**Chirality displacement** χ = u ⊕ v ∈ GF(2)⁶: the relative displacement between the two components.

**Shell** N = popcount(χ) ∈ {0,…,6}: the radial coordinate.

Given (c, χ), the state is recovered exactly:

```
u = c
v = c ⊕ χ
```

This is a bijection: 64 choices for c times 64 choices for χ equals 4096 = |Ω|. The shell determines the state count per shell:

```
|S_N| = 64 · C(6, N)
```

summing to 64 · 2⁶ = 4096.

The boundary anchor c is the holographic coordinate. The chirality displacement χ is the internal directional coordinate. The shell N is the finite radial coordinate.

### 1.5 Balanced shell and two-step uniformization

The shell distribution |S_N| = 64 · C(6, N) is symmetric under N ↔ 6 − N and is maximal at N = 3:

```
|S_3| = 64 · C(6,3) = 1280
```

The fraction N/6 = 1/2 at this maximum corresponds to equal contribution from both components. This is the discrete realization of the balanced occupancy point.

From any state s ∈ Ω, the 256² length-2 byte words distribute exactly uniformly over Ω: each of the 4096 states is reached by exactly 16 words. The future-cone measure at depth 2 is therefore exactly uniform. This is the native isotropization of the medium, achieved in exactly two steps.

---

## 2. The QuBEC Climate Object

### 2.1 Occupation measure

A one-cell climate at time t is an occupation measure on Ω:

```
p_t : Ω → [0,1],  Σ_{s ∈ Ω} p_t(s) = 1
```

The climate has three exact native marginals.

### 2.2 Chirality marginal

```
p_t^χ(χ) = Σ_{s : χ(s) = χ} p_t(s)
```

Represented by a 64-element integer histogram chi_hist64[0..63]. This is the primary native climate register.

### 2.3 Shell marginal

```
π_t(N) = Σ_{s : popcount(χ(s)) = N} p_t(s)
```

Represented by shell_hist7[0..6].

### 2.4 Gauge marginal

The byte family field partitions every byte into one of four K4 sectors. The gauge marginal tracks the empirical K4 sector occupation:

```
family_hist4[0..3]
```

### 2.5 State and process coordinates

The three marginals belong to two distinct coordinate types.

**State coordinates** describe where the cell is on Ω:
- boundary anchor c
- chirality displacement χ
- shell N

**Process coordinates** describe how the cell moves:
- gauge family ∈ K4, encoding the spinorial phase of each byte
- parity commitments, encoding frame-level trajectory structure
- depth-4 closure data, encoding accumulated phase over complete word cycles

These are complementary projections of one machine: where the cell is, how it moves, and what path it has taken.

---

# Part II. Native Transform Algebra

The QuBEC climate admits four native transform surfaces. Each is the unique exact harmonic basis of one exact chart of the medium. Together they form the transform algebra of the occupied QuBEC.

---

## 3. Chirality Spectral Transform

### 3.1 Definition

The chirality spectral transform is the Walsh-Hadamard transform on GF(2)⁶. For any function f : GF(2)⁶ → ℤ:

```
WHT(f)(u) = Σ_{χ ∈ GF(2)⁶} f(χ) · (−1)^{⟨u,χ⟩}
```

where ⟨u,χ⟩ = popcount(u ∧ χ) mod 2. The orthonormal matrix form is:

```
H(u,χ) = (−1)^{popcount(u ∧ χ)} / 8
```

### 3.2 Structural theorem

**Theorem.** The chirality register GF(2)⁶ is an additive group under ⊕. Byte transport on this register is exact XOR translation: χ′ = χ ⊕ q. By Pontryagin duality applied to the finite abelian group (GF(2)⁶, ⊕), the unique Fourier transform of this transport group is the Walsh-Hadamard transform. No other transform diagonalizes XOR-convolution exactly.

**Corollary.** The WHT is not a tool applied to the chirality register. It is the unique exact spectral dual of the transport law.

### 3.3 Transport and convolution law

For a byte-ensemble distribution ν(q) on q-charges, the spectral multiplier at mode u is:

```
φ(u) = Σ_q ν(q) · (−1)^{⟨u,q⟩}
```

Climate evolution in spectral space is:

```
A_{t+1}(u) = φ(u) · A_t(u)
```

Because transport is XOR translation, sequential composition is XOR-convolution:

```
(f ∗ g)(χ) = Σ_a f(a) · g(χ ⊕ a)
```

and the WHT diagonalizes this exactly:

```
WHT(f ∗ g)(u) = WHT(f)(u) · WHT(g)(u)
```

This is the exact composition law of climate transport in chirality space.

### 3.4 Fast execution form

The 64-point WHT executes via butterfly stages:

```
(x, y) → (x + y, x − y)
```

For n = 64:

- 6 stages
- 32 butterflies per stage
- 192 butterflies total
- 384 additions and subtractions

The transform requires no multiplications. All arithmetic is integer add and subtract.

### 3.5 Anisotropic factorization

When the byte ensemble drives the six chirality modes independently, with flip probability pᵢ on axis i, the spectral multiplier factorizes:

```
φ(u) = ∏_{i : uᵢ = 1} ηᵢ,   where ηᵢ = 1 − 2pᵢ
```

The isotropic case ηᵢ = η for all i gives φ(u) = η^{popcount(u)} = ηʳ for mode u of weight r.

---

## 4. Shell Radial Transform

### 4.1 Definition

The shell radial transform acts on the seven shell values N = 0,…,6. The Krawtchouk polynomials on H(6,2) are:

```
K_r(N) = Σ_{j=0}^{r} (−1)^j · C(N,j) · C(6−N, r−j)
```

The shell spectral transform is:

```
A(r) = Σ_{N=0}^{6} K_r(N) · π(N)
```

### 4.2 Structural theorem

**Theorem.** The shell index N = popcount(χ) defines the radial distance classes of the Hamming scheme H(6,2). By the theory of association schemes, the unique family of orthogonal polynomials invariant under all symmetries of H(6,2) is the Krawtchouk family. The Krawtchouk transform is therefore the unique exact radial harmonic basis of the shell quotient.

### 4.3 Radial transport law

For isotropic byte ensembles, where ν depends only on j = popcount(q), the shell dynamics diagonalize in the Krawtchouk basis:

```
A_{t+1}(r) = Λ(r) · A_t(r)
```

where Λ(r) is the exact radial eigenvalue determined by the ensemble.

### 4.4 Code-theoretic triple identity

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

### 4.5 Plancherel identity

The Plancherel identity on GF(2)⁶ relates energy in chirality space to energy in spectral space:

```
Σ_{χ=0}^{63} |f(χ)|² = (1/64) · Σ_{u=0}^{63} |WHT(f)(u)|²
```

For the occupation distribution p(χ), the left side is the inverse of the Rényi-2 effective support M₂. The degree of condensation in chirality space equals the degree of spectral excitation. The spectral profile reveals which of the six dipole-mode axes carry the condensation, information not recoverable from the scalar M₂ alone.

---

## 5. Gauge Character Transform

### 5.1 Definition

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

### 5.2 Structural theorem

**Theorem.** The family bits define a gauge sector in K4 ≅ GF(2)². By the character theory of finite abelian groups, the character table of K4 is the unique complete orthogonal decomposition of the gauge sector.

### 5.3 Gauge transport law

If gauge transport factorizes by the two family bits, with damping parameters ξ_A and ξ_B on each axis, the gauge spectral law is:

```
G_{t+1}(a,b) = ξ_A^a · ξ_B^b · G_t(a,b),   a,b ∈ {0,1}
```

### 5.4 Eight-axis damping system

The full QuBEC climate has eight independent damping axes under factorized anisotropy:

- six chirality axes η₁,…,η₆ from the six dipole modes
- two gauge axes ξ_A, ξ_B from the two K4 family bits

The combined spectral transport law is:

```
C_{t+1}(r, a, b) = ηʳ · ξ_A^a · ξ_B^b · C_t(r, a, b)
```

under isotropic chirality and factorized gauge. This is a tensor product of three independent spectral systems:

- radial (Krawtchouk, 7 modes)
- gauge-A (binary, 2 modes)
- gauge-B (binary, 2 modes)

Total spectral modes under shell reduction: 7 · 2 · 2 = 28.

---

## 6. Arithmetic Contraction Transform

### 6.1 K4 lattice matrix definition

For integer vectors q, k ∈ ℤⁿ, every signed 32-bit integer v admits the exact dyadic decomposition:

```
v = L(v) + B · H(v)
```

where B = 2¹⁶, L(v) is the signed low 16-bit value, and H(v) is the signed high 16-bit value. Define four contraction channels:

```
D₀₀(q,k) = ⟨L_q, L_k⟩     (carrier-carrier)
D₀₁(q,k) = ⟨L_q, H_k⟩     (carrier-gauge)
D₁₀(q,k) = ⟨H_q, L_k⟩     (gauge-carrier)
D₁₁(q,k) = ⟨H_q, H_k⟩     (gauge-gauge)
```

These assemble into the K4 lattice matrix:

```
M(q,k) = [[D₀₀, D₀₁],
           [D₁₀, D₁₁]]
```

The exact dot product is recovered by:

```
⟨q,k⟩ = D₀₀ + B·(D₀₁ + D₁₀) + B²·D₁₁
```

### 6.2 Three operational sectors

The four matrix entries carry three distinct operational roles.

**D₁₁: chiral alignment.** In the spinorial regime where H ∈ {−1, 0, +1}ⁿ, the gauge-gauge contraction uses signed support intersection:

```
D₁₁ = popcount(q⁺ ∧ k⁺) + popcount(q⁻ ∧ k⁻)
     − popcount(q⁺ ∧ k⁻) − popcount(q⁻ ∧ k⁺)
```

This counts aligned minus anti-aligned orientations across the support.

**D₀₁ and D₁₀: gauge action on the carrier.** The cross terms act as boolean control masks: where H = +1, L is preserved; where H = −1, L is sign-inverted; where H = 0, L is annihilated.

**D₀₀: carrier contraction.** The contraction of the low charts alone, with no gauge contribution.

### 6.3 Three computational regimes

The K4 lattice matrix admits three exact chart regimes determined by the high-chart occupancy of the data.

**Carrier regime.** H_q = H_k = 0. Only D₀₀ contributes. The sector budget is (1, 0, 0).

**Spinorial regime.** H ∈ {−1, 0, +1}ⁿ for both vectors. D₁₁ is computed by boolean support intersection. D₀₁ and D₁₀ are computed as signed masked actions.

**Dense regime.** |H| > 1 at some position. The K4 law is evaluated without boolean compression.

Regime selection is determined by the data, not by a precision parameter. All three regimes are exact.

### 6.4 Additive sector budget

Define:

```
E_carrier = |D₀₀|
E_cross   = B · (|D₀₁| + |D₁₀|)
E_align   = B² · |D₁₁|
```

The normalized sector budget satisfies:

```
ρ_carrier + ρ_cross + ρ_align = 1
```

This is an exact arithmetic identity describing how the total weighted magnitude distributes across the three operational sectors.

### 6.5 Dyadic radix identity

The arithmetic radix satisfies:

```
B = 2¹⁶ = 65536 = |Ω| · 16
```

where the factor 16 = |K4|² = 4². A radix shift from the low chart to the high chart corresponds to one Ω-manifold's worth of states scaled by the K4 gauge-square factor. The arithmetic radix is structurally commensurate with the manifold and gauge dimensions.

---

## 7. Unified Correspondence

### 7.1 Translation-convolution-spectrum theorem

**Theorem.** The following are equivalent charts of one transport structure:

1. Byte-induced chirality translation: χ′ = χ ⊕ q
2. XOR-convolution on GF(2)⁶: p_{t+1}(χ) = Σ_a ν(a) · p_t(χ ⊕ a)
3. Walsh-Hadamard spectral multiplication: A_{t+1}(u) = φ(u) · A_t(u)

### 7.2 Signed-support correspondence

**Theorem.** In the spinorial regime, where H ∈ {−1, 0, +1}ⁿ, the D₁₁ form and the Walsh character kernel measure the same alignment geometry under a fixed change of chart coordinates.

In the {−1, +1} encoding where bit 0 maps to +1 and bit 1 maps to −1, the inner product becomes:

```
⟨s_u, s_q⟩ = Σ_i (−1)^{uᵢ ⊕ qᵢ} = 6 − 2 · popcount(u ⊕ q)
```

The signed-support D₁₁ contraction and the Walsh character are dual coordinate expressions of the same correlation structure, related by chart-specific normalization and basis conventions.

### 7.3 Binary and ternary chart complementarity

The transport law is most natural in the binary chart of GF(2)⁶: state differences, chirality, XOR composition, Walsh characters, and code duality. The contraction law is most natural in the ternary chart {−1, 0, +1}: alignment counting, signed support intersections, and masked inversion or preservation.

These are not two competing systems. They are two exact alphabets of one medium, each native to a different computational act: transport in binary, contraction in ternary.

### 7.4 Exactness classes

All operations of the native transform algebra fall into one of three classes.

**Integer exact.** Exact over integer arithmetic with no approximation: byte stepping, q-charge extraction, chirality transport, unnormalized WHT, shell counting, parity commitments, K4 D₁₁ popcount contractions, and all horizon and shell observables.

**Dyadic exact.** Exact as rational values with denominators that are powers of two or small combinatorial factors: normalized WHT, K4 character normalization, shell probabilities, Krawtchouk-normalized coefficients, and climate occupation fractions. These remain exact and do not require floating-point semantics.

Dyadic exact values are represented as:

```
(numerator: int64, exponent: int8)
```

representing numerator · 2^{−exponent}. Normalization operations include:

```
WHT normalization:     right shift by 6  (divide by 64)
K4 character norm:     right shift by 2  (divide by 4)
```

When composing dyadic operations, exponents accumulate additively.

**WHT inverse scaling (normalization).** The 64-point WHT is its own inverse up to factor 64. The 16-point WHT used in the chi-gauge tile is its own inverse up to factor 16. Any implementation dividing by 4096 = 16^3 is applying three inverse scalings where only one is required, violating the Plancherel identity (Section 4.5), namely Σ_{χ=0}^{63} |f(χ)|² = (1/64) · Σ_{u=0}^{63} |WHT(f)(u)|², and the UV-IR conjugacy Q_G · m_a² = 1/2.

**Residual numerical.** Operators outside the exact quotient classes, non-native dense residuals, and approximate external backends. This class is not part of the native QuBEC transform algebra. It is the execution chart for whatever lies outside the exact quotient structure.

---

# Part III. Execution Consequences

---

## 8. Structured Operators and Quotient Hierarchy

### 8.1 Exact quotient classes

The QuBEC admits a hierarchy of exact operator classes ordered by increasing expressivity.

**Class 1: Shell-radial (7 parameters).** An operator W is shell-radial if it depends only on the shell index N = popcount(χ). It is specified by 7 Krawtchouk eigenvalues λ₀,…,λ₆.

**Class 2: Shell × gauge (28 parameters).** W depends on shell index and K4 gauge sector. This is the tensor product of the shell-radial class (7 modes) and the K4 character class (4 modes).

**Class 3: Chirality translation-invariant (64 parameters).** W commutes with all XOR translations on the chirality register. It is diagonal in the Walsh basis, specified by 64 spectral multipliers φ(u).

**Class 4: Chirality × gauge (256 parameters).** W is diagonal in the tensor product of the Walsh basis and the K4 character basis. Specified by 256 multipliers.

**Class 5: Generic (4096 parameters).** Operator has no translation, shell, or gauge invariance, therefore it is not diagonal in any native spectral basis. It is still evaluated exactly via the K4 lattice decomposition; no fallback to external arithmetic occurs.

### 8.2 Projection and defect

For any operator W and quotient class Q, the exact projection is:

```
P_Q(W) = component of W lying in class Q
D_Q(W) = W − P_Q(W)   (defect)
```

The structure capture ratio is:

```
SCR_Q(W) = ‖P_Q(W)‖_F / ‖W‖_F
```

where ‖·‖_F is the Frobenius norm.

For any operator W and chosen quotient Q, compute both terms always:

```
W · x = P_Q(W) · x + D_Q(W) · x
```

P_Q(W) · x uses the native transform diagonal (WHT, Krawtchouk, or K4Char). D_Q(W) · x uses the K4 lattice direct evaluation. SCR_Q(W) = ‖P_Q(W)‖_F / ‖W‖_F is a performance predictor, not a correctness gate; it must never disable the P_Q path.

---

## 9. Algorithms

### 9.1 n-step chirality evolution

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

### 9.2 Shell radial evolution

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

### 9.3 Structure analysis

Assigns a **dominant quotient chart label** to W for profiling, cache keys, and cost estimates. This routine does **not** define application semantics. Runtime multiplication is always Section 8.2: **W · x = P_Q(W) · x + D_Q(W) · x** with both terms evaluated in the native algebra. **SCR is not a gate:** a low SCR or a "generic" label never authorizes omitting **P_Q**, replacing it with a single dense matmul, or skipping the **K4 lattice** evaluation of **D_Q**.

```
Input:  W   operator matrix, shape (64, 64)
Output: structure_report

1. Test translation-invariance:
   W_circulant ← circulant reconstruction from W[0,:]
   if ‖W − W_circulant‖_F / ‖W‖_F < threshold:
       φ ← WHT of first row of W
       return {class: "translation-invariant",
               method: "wht-diagonal",
               eigenvalues: φ,
               scr: 1 − ‖D‖/‖W‖}

2. Test shell-radial structure:
   For each shell r, compute average W[i,j] over (i,j) in same shell class
   if off-shell-diagonal variance is small:
       λ ← 7 shell eigenvalues
       return {class: "shell-radial",
               method: "krawtchouk-diagonal",
               eigenvalues: λ,
               scr: ...}

3. Project onto χ × gauge basis (256 modes):
   Transform W into WHT ⊗ K4Char basis
   if diagonal norm / total norm > 0.8:
       return {class: "chi-x-gauge",
               method: "wht-k4char-diagonal",
               scr: ...}

4. return {class: "generic", notes: "apply Section 8.2; no correctness shortcut", scr: ‖P‖/‖W‖ for best attempted projection}
```

Analysis is performed once per operator and the result is cached. Cached labels inform implementation scheduling only; they do not change the exact sum **P_Q + D_Q**.

### 9.4 Horizon proximity detection

The **threshold** here is a **spectrum shape** tolerance (how close the empirical WHT coefficients are to binary {0, 64}). It is **not** an SCR value and **not** used for operator or block routing.

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

### 9.5 Anisotropy extraction

```
Input:  byte_ensemble   ν(b) for b = 0..255

Output: eta_vec   per-axis damping, float[6]

For each axis i in {1..6}:
    p_i ← Σ_{b : payload_bit_i(b) = 1} ν(b)
    eta_vec[i] ← 1 − 2 · p_i

return eta_vec
```

---

## 10. Symbolic Cost Semantics

### 10.1 Cost model

The total cost of a climate computation decomposes as:

```
C_total = C_extract + C_transform + C_apply + C_residual + C_memory
```

where:

- C_extract: cost to move the state into the chosen chart
- C_transform: cost of the native transform (WHT, Krawtchouk, K4Char, K4 lattice)
- C_apply: cost of diagonal or pointwise application
- C_residual: cost of evaluating the defect **D_Q** (K4 lattice path in the native algebra; external matmul is not its definition)
- C_memory: bytes moved and cache effects

### 10.2 Epistemic distinction

Three kinds of cost statements are distinguished.

**Symbolic cost.** Operation-count model on abstract arithmetic in each chart. Exact.

**Architecture-dependent execution cost.** Implementation cost on a concrete pipeline, dependent on memory hierarchy, vector width, and cast paths. Structural estimate.

**Measured benchmark cost.** Wall-time estimate from a target build. Empirical, and requires explicit citation of conditions.

The cost figures given below are symbolic unless stated otherwise.

### 10.3 Native transform costs

| Operation | Arithmetic | Multiplications |
|---|---|---|
| 64-point WHT | 384 add/sub | 0 |
| Krawtchouk7 | 42 multiply-add | 42 |
| K4Char4 | 12 add/sub | 0 |
| WHT + pointwise + inverse WHT | 832 total | 64 |

### 10.4 Single-step evolution: native vs dense

**Dense application of a 64×64 operator to a 64-element vector:**

```
C_apply:   64² = 4,096 multiply-accumulates
C_memory:  64 · 64 · 4 bytes = 16,384 bytes
```

**Native spectral application (translation-invariant operator):**

```
C_extract:    0  (histogram already in native form)
C_transform:  384 add/sub  (forward WHT)
C_apply:      64 multiplies
C_transform:  384 add/sub  (inverse WHT)
C_memory:     2 · 64 · 4 = 512 bytes

Total: 768 add/sub + 64 multiplies
```

**Symbolic ratio:** 4,096 / 832 ≈ 5× fewer arithmetic operations, 32× fewer bytes.

The advantage arises from diagonalization: the 64×64 matrix reduces to 64 independent scalars when the operator is translation-invariant.

### 10.5 n-step evolution: native vs dense

**Dense method via matrix exponentiation:**

```
Matrix exponentiation M^n via binary exponentiation:
   approximately log₂(n) matrix multiplies
   Each matrix multiply: 64³ = 262,144 multiply-accumulates

Final matrix-vector product: 64² = 4,096 multiply-accumulates
```

**Native spectral method (Algorithm 9.1):**

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

### 10.6 Multi-cell batch

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

## 11. Lowering and Interoperability

### 11.1 The 64-wide lowering grain

Width 64 is the canonical grain at which the following structures coincide:

- horizon cardinality |H| = 64
- chirality register dimension |GF(2)⁶| = 64
- self-dual mask code size |C₆₄| = 64
- 64-point WHT dimension
- 64-byte L1 cache line (6-bit offset)

This is the natural interoperability grain between external tensors of arbitrary width and the native QuBEC computation surfaces.

### 11.2 Block tiling of external tensors

An external tensor of width d tiles into native blocks as:

```
d = 64k        → k exact blocks
d = 64k + r    → k blocks + 1 residual block, padded to 64
```

Each 64-wide block projects onto the chirality register basis. A transformer layer of width 768 becomes 12 blocks. A width-4096 layer becomes 64 blocks. Each block is individually addressable by the exact operator quotient classes of Section 8.

### 11.3 External tensor ingestion

```
Input:  W       dense weight matrix, shape (rows, d)

Output: block_registry

1. Pad d to next multiple of 64 if d mod 64 ≠ 0
2. For each block b in range(d // 64):
       W_block ← W[:, b·64 : (b+1)·64]
       report  ← analyze_operator(W_block)   (Algorithm 9.3)
       block_registry[b] ← (W_block, report)

return block_registry
```

### 11.4 Hybrid block application

Correct block application is the Section 8.2 identity for the block operator **W_block**. The registry **report** from Algorithm 9.3 only names which transform diagonalizes the **structured** part **P_Q(W_block)** when **W_block** lies exactly in that class; it must **not** be read as "if label is generic then **W_block @ x** is the definition of the product." For every label, compute **y_b = P_Q(W_block) · x_block + D_Q(W_block) · x_block**: native diagonal application for **P_Q**, K4 lattice direct evaluation for **D_Q**. Optional forwarding of a **stored** residual to an external backend is a deployment/storage choice; it is not a substitute for that sum when defining **W · x**.

```
Input:  block_registry, x   input vector of width d

Output: result vector

1. For each block b in block_registry:
       x_block <- x[b*64 : (b+1)*64]
       (W_block, report) <- block_registry[b]    // as built in Algorithm 11.3

       y_b <- native_section_8_2_apply(W_block, x_block, report)
       // native_section_8_2_apply implements P_Q(W_block)*x_block + D_Q(W_block)*x_block
       // using WHT / Krawtchouk / WHT+K4Char for the P_Q branch that matches W_block,
       // and K4 lattice evaluation for D_Q. Do not branch on report.scr or SCR thresholds.

2. return concatenate(y_b for all b)
```

A literal **if class ... else matmul** sketch is a **cost model illustration only** when all mass lies in one quotient; it is not the semantic specification of multiplication.

### 11.5 Derived application examples

The following are illustrative consequences of the polar decomposition and the 64-wide lowering, not part of the core formalism.

**KV embedding encoding.** A KV embedding of dimension d tiles into 64-wide blocks. Each block has canonical polar coordinates (c, χ, N) from Section 1.4. With a float16 radius r per block capturing the L2 magnitude, the storage per block is:

```
c: 6 bits
N: 3 bits
χ: 6 bits
r: 16 bits (float16)
Total: 31 bits per block
Uncompressed: 64 · 16 = 1024 bits per block
```

The compression ratio is approximately 33×. Reconstruction recovers the approximate embedding from (c, χ, r) using the shell representative scaled by r.

**Native attention approximation.** For query q and key k encoded in polar coordinates, a native approximation to the dot product attention score uses:

```
chirality_distance ← popcount(χ_q ⊕ χ_k)
shell_similarity   ← 6 − chirality_distance
anchor_alignment   ← popcount(c_q ⊕ c_k)
score              ← r_q · r_k · (shell_similarity / 6) · (1 − anchor_alignment / 64)
```

This gives a closed-form attention approximation without full embedding reconstruction.

---

# Part IV. Closure Geometry

---

## 12. Aperture Constants and Transform Algebra

### 12.1 Aperture constants

The CGM constants governing closure geometry are:

```
m_a    = 1 / (2√(2π))    observational aperture
δ_BU                      BU dual-pole monodromy
ρ      = δ_BU / m_a       closure ratio
Δ      = 1 − ρ            aperture gap  (≈ 0.0207)
Q_G    = 4π               quantum gravity invariant
```

### 12.2 Observational aperture as root extraction

The observational aperture m_a equals the derivative of √x evaluated at x = 2π:

```
d/dx (√x)|_{x=2π} = 1/(2√(2π)) = m_a
```

The central normalization identity is:

```
Q_G · m_a² = 1/2
```

This states that the product of the full solid angle with the squared root-extraction rate at the phase horizon equals the half-integer 1/2, connecting the observational geometry to the SU(2) double-cover structure.

### 12.3 Polar role forced by the aperture

Because Δ > 0, the system is not absolutely closed. A fully closed system (Δ = 0) would permit characterization by a scalar magnitude alone. The irreducible aperture residue Δ > 0 forces the representation to include both the radial coordinate (shell N) and the directional coordinate (chirality χ). Neither is eliminable.

This is the structural reason the canonical decomposition of Ω into (c, χ, N) has three components rather than one. The shell N is the radial magnitude. The chirality χ is the residual orientation. Both are required by the geometry of non-zero aperture.

### 12.4 Depth-4 aperture quantization

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

### 12.5 Unified defect concept

The transform algebra and the closure geometry share a common structural concept: an exact finite remainder from an ideal closure condition.

**Closure defect.** The aperture gap Δ is the remainder when the BU monodromy falls short of the aperture scale. It is the structural reason the medium has both radial and directional coordinates.

**Anisotropy defect.** When the byte ensemble drives the six chirality axes unequally, the climate departs from the isotropic shell distribution. The anisotropy vector (η₁,…,η₆) measures this defect from the isotropic ideal.

**Quotient defect.** When an operator W does not lie exactly in a quotient class Q, the defect D_Q(W) = W − P_Q(W) measures the departure from exact structured form. **D_Q** is still evaluated **exactly** in the **K4 lattice** chart; "dense" in cost tables means high multiply-accumulate count, not permission to drop native QuBEC semantics. Optional ambient matmul is an external offload of a **stored** remainder, not the theory's definition of **D_Q(W) · x**.

All three are structurally the same kind of object: a finite remainder from an exact closure ideal, quantifiable, and segregable from the exact part.

---

## 13. Implementation Correspondence

The transform algebra formalized in this document corresponds to three layers of the implementation stack.

**Semantic layer.** The Python `src` surface defines the exact semantics: kernel transition law, chirality transport, exact Walsh matrix, exact Krawtchouk coefficients, and all state and transport observables. All transforms defined in this document have their normative reference in that layer.

**Native execution layer.** The GyroLabe native surfaces provide hardware-near execution of the kernel-exact operations: byte stepping, q-map extraction, chirality distances, signature scans, and the 64-point WHT. The fast WHT is the butterfly realization of the chirality spectral transform defined in Section 3. The packed tensor surfaces realize the K4 arithmetic contraction chart of Section 6 at the 64-wide grain.

**Empirical climate layer.** The GyroGraph rolling memories (chi_hist64, shell_hist7, family_hist4) are the implementation-level empirical summaries of the one-cell climate marginals defined in Section 2. The SLCP spectral64 field is the WHT of chi_hist64, which is the empirical chirality spectral transform evaluated on the rolling observation window.

---

## Unified Statement

The occupied QuBEC is one finite exact computational medium. Its kernel semantics, climate marginals, spectral transforms, shell harmonics, gauge decomposition, and arithmetic contraction charts are exact coordinate systems on one machine.

The Walsh-Hadamard transform is the unique exact spectral chart of chirality transport, forced by Pontryagin duality on GF(2)⁶. The Krawtchouk transform is the unique exact radial chart of shell transport, forced by the association scheme H(6,2). The K4 character transform is the unique exact gauge chart of family-phase occupation, forced by the character theory of GF(2)². The K4 lattice matrix is the exact arithmetic chart of structured contraction, with the D₁₁ signed-support form as the ternary face of the same correlation geometry that the WHT expresses spectrally.

Binary XOR governs transport. Ternary signed-support governs contraction. Both are exact alphabets of one medium.

The canonical polar decomposition (c, χ, N) is the native coordinate system of Ω. The shell maximum at N = 3 is the discrete balanced-occupancy point. Two-step uniformization is the native isotropization. These are not analogies to continuous structures. They are the exact finite structures of which continuous analogs are the limiting shadows.

The operator quotient hierarchy maps any operator to its exact structured component plus a finite defect. **P_Q** routes through the appropriate native transform diagonal; **D_Q** routes through the **K4 lattice** direct evaluation. Both are mandatory for **W · x**. SCR compares their relative **symbolic** weights for **performance prediction only**; it does not decide whether the native **P_Q** path runs.

The aperture gap Δ forces the coexistence of radial and directional coordinates. The depth-4 aperture quantization 48Δ ≈ 1 connects the continuous closure geometry to the discrete byte-horizon structure. The unified defect concept connects closure geometry, anisotropy, and quotient structure through one common form: a finite remainder from an exact ideal.