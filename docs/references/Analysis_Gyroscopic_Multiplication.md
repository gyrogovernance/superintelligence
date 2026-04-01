

# Gyroscopic Multiplication: Independence Roots and Aperture Reproducibility

**Author:** Basil Korompilias
**Date:** March 2026
**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

---

## Abstract

This document examines the mathematical structure of multiplication through the lens of the Common Governance Model (CGM). The analysis establishes that multiplication is the orthogonal case of bilinear spanning, that root extraction recovers a shared measure from a higher-degree closure, and that the CGM aperture parameter arises as the exact derivative of the square root function at a full phase horizon.

The arithmetic realization of these identifications is developed through the K4 lattice matrix: the canonical 2 x 2 decomposition of integer dot products into carrier, gauge-action, and chiral-alignment sectors. The scalar case satisfies exact rank-1 factorization (the common-source condition), while the vector case produces a nonzero chart defect decomposable via Cauchy-Binet into chart commutators that measure scale inhomogeneity across positions. The depth hierarchy of this decomposition traces the CGM constraint progression from common source through non-commutativity to balanced closure.

These identifications connect the CGM geometric invariants to the classical theory of roots of unity, continued fractions, Gram determinants, Hilbert space norms, quaternionic orientation spaces, and two-circle intersection geometry. Cross-domain resonances are examined in the transition from integrability to chaos in Hamiltonian dynamical systems, where a universal critical exponent of one half governs the onset of non-integrable behavior, and in the lemon billiard family, where the CGM monodromy constant appears as the shape parameter producing a uniquely balanced mixed-type phase space.

Results are stratified by epistemic status: exact mathematical results, structural correspondences between layers, and phenomenological observations generating falsifiable hypotheses.

---

## Epistemic Framework

This document follows the epistemic stratification of the CGM paper:

**Exact.** Results that follow from definitions and standard mathematical theorems. Their negation entails mathematical contradiction.

**Structural correspondence.** Identifications between objects at different layers of the formalism (abstract, arithmetic, geometric) that share the same algebraic structure. These are not conjectures but they are not proofs of physical identity. They identify shared form.

**Phenomenological.** Numerical agreements or cross-domain parallels that generate falsifiable hypotheses but have not been derived from first principles within the framework. These are clearly labeled.

---

## Part I: Foundations

### 1. Definitions

#### 1.1 Orthogonality

Two directed quantities are orthogonal when operating along one produces zero progress along the other. Formally, two vectors u and v in an inner product space are orthogonal when their inner product vanishes:

⟨u, v⟩ = 0

Orthogonality ensures that the two directions provide completely independent information. Motion along u reveals nothing about v, and motion along v reveals nothing about u.

#### 1.2 Dimension

A dimension is an irreducible degree of operational freedom. It is the capacity of a single freedom to be placed in a way that cannot be reduced to placements already given.

The dimension of a space is the maximum number of mutually orthogonal directions it supports. Each orthogonal direction represents a placement of operational freedom that is genuinely independent of all others.

The emergence of multiple dimensions is the self-differentiation of one operational freedom through orthogonal placement. When a single operational freedom is placed in relation to itself and the resulting placement cannot be collapsed back to the original, a dimension has emerged.

#### 1.3 Root

A root of a closure law is a value that satisfies the law. The square root of x is a number y satisfying y² = x. The nth root of unity is a number z satisfying zⁿ = 1. More generally, a root of a polynomial p is a value y satisfying p(y) = 0.

In every mathematical tradition that independently developed the concept, the same word was chosen: root (Arabic jaḏr, Latin radix, Sanskrit mūla). The root is the shared measure that, when placed in a higher-degree closure law, satisfies it.

A root is not an originating entity. It is the measure recoverable from a closure. The root does not produce the square. The root is what the square yields when the closure law is inverted.

#### 1.4 Square

A square is the closure obtained when one measure is reproduced in two orthogonal placements. The two placements are distinct (they occupy different directions) but commensurable (they carry the same magnitude). The closure is the area spanned between them.

The squareness (equal sides) means the two orthogonal placements carry the same measure. The root is that common measure.

#### 1.5 Common Source

The Common Source, as formalized in the Common Governance Model, is the shared traceability condition under which distinct operational placements remain commensurable. It is a unary condition on relation, stating that operational structure must trace to a shared origin. It is not an entity prior to relation. It is the condition that relation preserves.

This condition is what makes multiplication well-defined: two factors can be multiplied because they share a common system of measure. Two directions can span an area because they both carry magnitudes defined within the same inner product structure. The Common Source is the reason the product is meaningful, not the object that produces the product.

---

### 2. Multiplication as Orthogonal Bilinear Spanning

#### 2.1 The Gram determinant

For two vectors u and v in an inner product space, the squared area of the parallelogram they span is given by the Gram determinant:

Area(u, v)² = det G(u, v)

where G(u, v) is the Gram matrix:

```
G(u, v) = [ ⟨u, u⟩   ⟨u, v⟩ ]
           [ ⟨v, u⟩   ⟨v, v⟩ ]
```

Expanding:

Area(u, v)² = ‖u‖² ‖v‖² − ⟨u, v⟩²

This is the exact relation between bilinear spanning (area production) and orthogonality (independence of directions).

#### 2.2 The orthogonal case

When u and v are orthogonal (⟨u, v⟩ = 0), the Gram determinant simplifies to:

Area(u, v) = ‖u‖ · ‖v‖

Multiplication of magnitudes is the orthogonal case of bilinear spanning. Area production reduces to magnitude multiplication precisely when the two spanning directions are independent.

A square is the further special case in which the two spanning measures are equal: ‖u‖ = ‖v‖, yielding Area = ‖u‖².

#### 2.3 The wedge product and chirality

The exterior (wedge) product captures oriented area directly:

u ∧ v

with norm ‖u ∧ v‖ = Area(u, v). Under orthogonality, ‖u ∧ v‖ = ‖u‖ · ‖v‖.

The wedge product is antisymmetric:

u ∧ v = −(v ∧ u)

Reversing the order of factors negates the product. This antisymmetry encodes chirality: the two possible orderings of u and v produce the same magnitude of area with opposite orientations. The two square roots of a positive number (positive and negative) correspond to the two orientations of the parallelogram spanned by a measure placed in two orthogonal roles.

#### 2.4 Two square roots from chirality

Every positive real number x has exactly two square roots: +√x and −√x. This follows from the antisymmetry of the wedge product. The area spanned by two directions can be oriented clockwise or counterclockwise. Both orientations produce the same magnitude (the same positive number x) but differ in sign. The squaring map y → y² sends both +y and −y to y², erasing the orientation information.

Recovering the root from the square encounters an inherent ambiguity: the square does not record which orientation produced it. This sign ambiguity is the arithmetic expression of chirality.

#### 2.5 The principal root as a chart choice

For positive real numbers, the convention of choosing the positive square root as principal is a chart choice on the real line, selecting one branch of a multivalued function. For complex square roots, the choice of principal value requires a branch cut in the complex plane.

The underlying root structure is multivalued and chiral. The principal root is a convention that selects one branch, not a structural privilege of one root over another.

---

### 3. Three Dimensions and the Product as Direction

#### 3.1 The Hodge dual in three dimensions

In general dimensions, the product of two independent directions is a bivector (an oriented area element). In three dimensions, and only in three dimensions in the standard setting, every bivector is Hodge-dual to a unique vector normal to the plane:

*(u ∧ v)

This is the mathematical basis of the vector cross product:

u × v = *(u ∧ v)

In three dimensions, the product of two independent directions yields not only an area but also a third direction orthogonal to both. The product of two freedoms canonically becomes a third freedom.

#### 3.2 Why three dimensions are special for multiplication

Two independent directions are required to construct a product (the factors). In dimensions greater than three, the bivector u ∧ v does not correspond to a unique normal direction. In dimensions less than three, there is no room for two independent factors and an observer. In exactly three dimensions:

- Two orthogonal placements span an area
- The area determines a unique third direction (the normal)
- This third direction serves as the observational axis from which the product can be perceived as an enclosed quantity

Three dimensions are the first and only standard dimension in which the product of two independent directions canonically produces an independent third direction.

#### 3.3 The quaternion multiplication table

The quaternion units i, j, k satisfy:

i² = j² = k² = −1

ij = k,  jk = i,  ki = j

with sign reversal under reversed order (ji = −k, etc.). This is the algebraic realization of the Hodge dual in three dimensions: the product of two orthogonal unit directions yields the third, with chirality encoded by sign.

#### 3.4 The orientation space of roots of −1

In the quaternion algebra, the equation q² = −1 has the solution set:

{ai + bj + ck : a² + b² + c² = 1}

This is the unit 2-sphere S² in the imaginary quaternion space. Each point on this sphere is a unit imaginary quaternion that, applied twice, produces complete reversal (multiplication by −1).

The total solid angle of this orientation space is:

∫_{S²} dΩ = 4π

This is the CGM quantum gravity invariant Q_G. The quantity 4π is the total angular extent of admissible root orientations in three dimensions.

Each point on the sphere is also an oriented 2-plane normal: the direction perpendicular to the plane in which the quarter-turn rotation occurs. This connects the quaternionic root structure back to the Hodge dual: each root of −1 is simultaneously a rotation axis and a bivector normal.

---

### 4. Root Extraction in Inner Product Spaces

#### 4.1 The norm as root of a quadratic form

In any Hilbert space, the norm of a vector v is defined by:

‖v‖ = √⟨v, v⟩

The inner product ⟨v, v⟩ is a quadratic observable (quadratic in v). The norm ‖v‖ is the linear quantity recovered from this quadratic observable by root extraction.

Root extraction is the standard mechanism by which inner product geometry recovers amplitude from self-interaction. The square root maps a quadratic form (the inner product, the intensity, the probability) back to a linear quantity (the amplitude, the norm, the magnitude).

#### 4.2 Variance and standard deviation

The same root-extraction structure appears in statistics and dynamical systems theory. Given a random variable ω with mean ⟨ω⟩, the variance is:

Var(ω) = ⟨ω²⟩ − ⟨ω⟩²

and the standard deviation is:

σ = √Var(ω)

The standard deviation recovers a linear-scale quantity from a quadratic dispersion measure. The root-mean-square (RMS) value:

ω_rms = √⟨ω²⟩

recovers linear scale from a second moment.

In the study of dynamical phase transitions (Section 13), the order parameter governing the transition from integrability to chaos is precisely such an RMS quantity. The observable that measures the onset of chaos is built by extracting a square root from a quadratic dispersion.

#### 4.3 Tensor products and multiplicative dimension

For finite-dimensional Hilbert spaces H and K:

dim(H ⊗ K) = dim(H) × dim(K)

Dimensions multiply under independent composition. If H supports m orthogonal freedoms and K supports n orthogonal freedoms, their independent composition supports mn orthogonal freedoms.

Multiplication counts the number of independent pairings generated by reproducible orthogonal placement. Each pair (one freedom from H, one from K) constitutes an independent combined freedom in the tensor product.

---

## Part II: CGM Root Identities

### 5. Exact CGM Invariants as Root Identities

#### 5.1 The aperture as the derivative of the square root

The CGM observational aperture parameter is defined as:

m_a = 1 / (2√(2π))

Numerically, m_a ≈ 0.199471. This is an exact closed-form constant, derived from the depth-four balance condition of the CGM framework.

The square root function f(x) = √x has derivative:

f'(x) = 1 / (2√x)

Evaluating at x = 2π (one complete phase cycle):

f'(2π) = 1 / (2√(2π)) = m_a

The aperture parameter is the rate of change of the square root function at the boundary of a full phase horizon. It measures how rapidly the root changes per unit of observable at the point where the phase completes a full cycle.

This identification is exact and algebraic.

#### 5.2 The normalization condition

The CGM normalization is:

Q_G · m_a² = 1/2

where Q_G = 4π is the total solid angle. Substituting m_a = f'(2π):

4π · [f'(2π)]² = 1/2

This connects the total observational solid angle (the quaternionic root orientation space) to the squared sensitivity of the root extraction process at one full phase cycle, with the value 1/2 reflecting the SU(2) double-cover structure (spin-1/2).

Rearranging:

m_a = √(1 / (2Q_G)) = √(1 / (8π))

The aperture is the square root of the ratio between the half-integer (the SU(2) spin quantum number) and the complete observational horizon (the total solid angle). The aperture itself is obtained by root extraction.

#### 5.3 The orthogonality threshold

The CGM Common Source threshold is s_p = π/2, the right angle. This is the minimal phase angle establishing directional distinction between the two fundamental transitions.

Geometrically, the right angle is the condition of full orthogonality: two directions at π/2 have zero projection onto each other, ensuring complete independence. Root extraction requires this angle. Euclid's geometric construction of √a (Elements, Propositions II.14 and VI.13) proceeds by inscribing a triangle in a semicircle. By Thales' theorem, the inscribed angle is necessarily π/2. Without this right angle, the similar-triangle argument yielding h² = ab fails, and the root cannot be extracted.

The coincidence between the CGM orthogonality threshold and the Thales angle is structurally necessary. The extraction of a root (the recovery of a shared measure from its closure) geometrically requires that the two placements be fully orthogonal.

#### 5.4 The chirality-aperture relation

The CGM framework identifies:

s_p / m_a² = (π/2) / (1/(8π)) = 4π²

The orthogonality threshold, when normalized by the squared aperture, yields 4π². This factor appears in the optical conjugacy relation connecting ultraviolet and infrared physics:

E_i^UV · E_i^IR = (E_CS · E_EW) / (4π²)

The geometric dilution factor 4π² connecting scales across the full observational range is the ratio of the right angle to the squared root-extraction rate.

#### 5.5 The geometric mean action and the cube root of unity

The CGM geometric mean action is:

S_geo = m_a · π · √3/2

The factor √3/2 is the imaginary part of the non-trivial cube roots of unity. The cube roots of 1 are:

1,  (−1 + i√3)/2,  (−1 − i√3)/2

The imaginary component ±√3/2 measures the maximal orthogonal extension achievable by a third root of unity relative to the real axis. It is the altitude of the equilateral triangle inscribed in the unit circle.

The geometric mean action therefore combines three quantities:

- m_a: the root-extraction rate at 2π (the aperture)
- π: the half-cycle phase
- √3/2: the three-dimensional orthogonal extension (from the cube root structure)

The gravitational coupling scale follows:

ζ = Q_G / S_geo = 4π / (m_a · π · √3/2) = 16√(2π/3)

---

### 6. Roots of Unity and Balanced Closure

#### 6.1 Definition

The nth roots of unity are the complex numbers z satisfying zⁿ = 1:

z_k = exp(2πik/n) = cos(2πk/n) + i sin(2πk/n),  k = 0, 1, …, n−1

These n numbers are equally spaced on the unit circle in the complex plane, forming a cyclic group under multiplication.

#### 6.2 The summation law

The sum of all nth roots of unity vanishes for n > 1:

SR(n) = Σ_{k=0}^{n−1} z_k = 0    for n > 1

This is a consequence of Vieta's formulas: the sum of the roots of zⁿ − 1 equals the coefficient of z^{n−1}, which is zero for n > 1. Geometrically, the roots are symmetrically distributed on the unit circle, so their centroid is the origin.

The vanishing sum is the algebraic expression of balanced closure. When all orientations are fully expressed around a complete cycle, they cancel to zero. The complete cycle contains all the structure, and that structure sums to exact neutrality.

In the CGM framework, this corresponds to the depth-four balance condition: the depth-four commutator vanishes in the S-sector, meaning all accumulated phase differences neutralize over a complete operational loop.

#### 6.3 Orthogonality of roots

The roots of unity satisfy an orthogonality relation:

Σ_{k=1}^{n} z̄^{jk} · z^{j'k} = n · δ_{j,j'}

The n phase modes generated by a primitive nth root form an orthogonal basis for the space of n-periodic sequences. Orthogonality is realized by the vanishing of their inner products under summation.

The n × n matrix U with entries U_{j,k} = n^{−1/2} · z^{jk} defines a discrete Fourier transform, and the orthogonality relation ensures that U is unitary. This is the foundation of Fourier analysis: decomposition of periodic structure into orthogonal modes.

#### 6.4 Cube roots and three-dimensional structure

For n = 3, the non-trivial cube roots of unity have real part −1/2 and imaginary part ±√3/2. The cyclotomic polynomial is:

Φ₃(z) = z² + z + 1

The cube roots form an equilateral triangle inscribed in the unit circle. The altitude of this triangle, √3/2, is the maximal orthogonal extension and appears throughout the CGM framework as the three-dimensional projection factor.

#### 6.5 Fourth roots and depth-four closure

For n = 4, the cyclotomic polynomial is:

Φ₄(z) = z² + 1

The primitive fourth roots of unity are ±i, the square roots of −1 in the complex numbers. The fourth roots {1, i, −1, −i} form the vertices of a square on the unit circle.

This fourth-order closure corresponds to the CGM depth-four balance: the depth-four commutator of the two operational transitions vanishes, achieving cyclic return. The fourth root structure provides the minimal finite cyclic group that supports both chirality (sign reversal at z² = −1) and closure (return to identity at z⁴ = 1).

---

### 7. Monodromy and Non-Closure

#### 7.1 Periodic continued fractions

Lagrange established (c. 1780) that the continued fraction expansion of the square root of any non-square positive integer is periodic:

```
√2 = [1; 2, 2, 2, …]
√3 = [1; 1, 2, 1, 2, …]
√5 = [2; 4, 4, 4, …]
```

The repeating block never terminates (the square root is irrational) but cycles with a fixed period. This periodic non-closure is the arithmetic form of monodromy. The system wraps around its repeating block, returning to the same pattern without achieving exact closure.

#### 7.2 The CGM monodromy defect

The CGM dual-pole monodromy defect is δ_BU ≈ 0.195342 radians, the phase accumulated by a depth-four cycle that almost closes but retains a small residual. The closure ratio is:

ρ = δ_BU / m_a ≈ 0.9793

The cycle closes to 97.93%, with a 2.07% aperture gap:

Δ = 1 − ρ ≈ 0.0207

This gap, like the irrationality of √2, prevents exact closure while maintaining a precise, repeating geometric structure.

#### 7.3 The Riemann surface of the square root

For the complex square root, the function z → √z is multivalued: each nonzero z has two square roots. To make the function single-valued and continuous, one passes to a Riemann surface with two sheets, connected at a branch point at z = 0.

Traversing a closed loop around the branch point once produces:

√z → −√z

The two roots are exchanged. One circuit produces non-closure with sign reversal. A second circuit restores closure:

−√z → √z

The square-root Riemann surface provides a canonical minimal monodromy model: reproducible non-closure with finite return depth (depth 2). For cube roots, the Riemann surface has three sheets, and one circuit multiplies the root by exp(2πi/3), a primitive cube root of unity. Three circuits restore closure.

#### 7.4 Heron's method: root recovery through balance

Heron's method (the Babylonian method) for computing √a is:

x_{n+1} = (1/2)(x_n + a/x_n)

If x_n is an overestimate of √a, then a/x_n is an underestimate. Their arithmetic mean yields a better estimate. The method converges quadratically: the number of correct digits roughly doubles with each iteration.

The root is recovered through a balance state between excess and deficiency. An overestimate and its reciprocal underestimate converge to the common measure through repeated averaging. The iteration never reaches the root in finite steps (for irrational roots), maintaining an aperture that diminishes with each cycle but never vanishes.

This is structurally identical to the CGM balanced closure principle: the shared measure (the root) is inferentially recoverable from a balanced state between complementary approximations.

---

## Part III: Arithmetic Realization

### 8. Lattice Multiplication

#### 8.1 The lattice principle

Lattice multiplication computes a product by arranging all digit-pair products on a rectangular grid. Each cell holds one local product. Each diagonal groups all contributions to one output place value. The result is read by summing along diagonals and propagating carries.

For two numbers x and y written in base B:

```
x = Σ x_i B^i
y = Σ y_j B^j
```

the product is:

```
xy = Σ_i Σ_j x_i y_j B^{i+j}
```

Each term x_i y_j occupies one cell. Each diagonal collects all terms sharing the same exponent i + j. The lattice is a different organization of the same law, one that makes place-value geometry explicit.

#### 8.2 Historical lineage

Lattice multiplication appears across multiple mathematical traditions: Indian (Kapat-sandhi, in commentary on the 12th century Lilavati of Bhaskara II), Arabic (Ibn al-Banna al-Marrakushi, Talkhis a'mal al-hisab, late 13th century), European (anonymous Latin treatise from England, c. 1300; Treviso Arithmetic, 1478; Pacioli, Summa de arithmetica, 1494), Chinese (Wu Jing, Jiuzhang suanfa bilei daquan, 1450), and Ottoman (Matrakci Nasuh, Umdet-ul Hisab, 16th century). The same structural principle informed mechanical aids such as Napier's bones and Genaille-Lucas rulers.

The evidence suggests either transmission across regions or independent development in multiple cultures. The method belongs to no single tradition.

#### 8.3 Structural properties

Lattice multiplication exposes five structural properties that remain important in modern computation:

1. Multiplication decomposes into independent local products.
2. Place value is determined by grid position, not by execution order.
3. Carries are localized between neighboring diagonals.
4. Cells can be filled in any order (parallelism is natural).
5. The method generalizes to any radix.

The Comba multiplication technique used in multi-precision arithmetic is the same organizational move: group partial products by output column rather than processing row by row.

#### 8.4 Binary bitplanes

In binary, each integer x is written as:

```
x = Σ x_i 2^i,  x_i ∈ {0, 1}
```

The product formula becomes lattice multiplication with bits as digits. The binary version proceeds: split each operand into bitplanes, compute all plane-pair interactions using AND, count overlaps using POPCNT, weight each pair by 2^{i+j}, and sum.

#### 8.5 The Nikhilam connection

Nikhilam (from Vedic mathematics, the sutra Nikhilam Navatashcaramam Dashatah) is a near-base multiplication method. For base B:

```
x = B + a
y = B + b
xy = B² + B(a + b) + ab
```

The expensive work shifts from one large multiply to a smaller residual multiply (ab) plus linear base-scaled terms. This is structurally a two-digit lattice with radix B. When B = 65536, the Nikhilam decomposition becomes the dyadic chart used in gyroscopic multiplication.

---

### 9. The K4 Lattice Matrix

#### 9.1 The dyadic chart

Every signed 32-bit integer v admits a unique decomposition:

```
v = L(v) + B · H(v)
```

where B = 2¹⁶ = 65536, L(v) is the signed low 16-bit value (in [−2¹⁵, 2¹⁵ − 1]), and H(v) is the signed high 16-bit value (in [−2¹⁵, 2¹⁵ − 1]).

This decomposition is exact and unique.

The low chart L captures the carrier content. The high chart H captures the gauge-scale content. This is a two-digit lattice decomposition with radix 65536, the modern algebraic form of the lattice principle.

#### 9.2 The four contraction channels

For vectors q, k ∈ ℤⁿ, define:

```
D₀₀(q, k) = ⟨L_q, L_k⟩       carrier-carrier
D₀₁(q, k) = ⟨L_q, H_k⟩       carrier-gauge
D₁₀(q, k) = ⟨H_q, L_k⟩       gauge-carrier
D₁₁(q, k) = ⟨H_q, H_k⟩       gauge-gauge
```

These assemble into a 2 × 2 matrix:

```
M(q, k) = [ D₀₀  D₀₁ ]
           [ D₁₀  D₁₁ ]
```

The ordinary dot product is recovered by the radix projection:

```
⟨q, k⟩ = (1, B) · M(q, k) · (1, B)ᵀ
       = D₀₀ + B(D₀₁ + D₁₀) + B²D₁₁
```

This identity is exact for every int32 dot product. The K4 index set {00, 01, 10, 11} is (ℤ/2)².

#### 9.3 Three operational roles

The four entries carry three distinct operational roles.

**D₁₁: chiral alignment.** When H takes values in {−1, 0, +1}, the gauge-gauge contraction is computed by signed support intersection:

```
D₁₁ = popcount(q⁺ ∧ k⁺) + popcount(q⁻ ∧ k⁻)
     − popcount(q⁺ ∧ k⁻) − popcount(q⁻ ∧ k⁺)
```

This is aligned-minus-anti-aligned counting, the same algebraic form as chirality measurement on oriented pairs.

**D₀₁ and D₁₀: gauge action on the carrier.** In the spinorial regime, the cross terms act as boolean control masks over the carrier content: where H = +1, L is preserved; where H = −1, L is sign-inverted; where H = 0, L is annihilated.

**D₀₀: carrier contraction.** The contraction of the low charts alone, with no gauge contribution.

#### 9.4 The additive sector budget

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

This is an exact arithmetic identity describing how the total weighted magnitude distributes across the three operational sectors.

#### 9.5 Three computational regimes

The K4 lattice matrix admits three exact chart regimes determined by the high-chart occupancy.

**Carrier regime.** H_q = 0 and H_k = 0.

```
M(q, k) = [ D₀₀  0 ]
           [ 0    0 ]
```

Only D₀₀ contributes. The budget is (1, 0, 0). This is the Nikhilam regime where all residuals vanish.

**Spinorial regime.** H ∈ {−1, 0, +1}ⁿ for both vectors.

D₁₁ is realized by boolean support intersection (AND + POPCNT on sign masks). D₀₁ and D₁₀ are realized as signed masked actions. All four cells are computed exactly with compressed boolean arithmetic.

**Dense regime.** |H| > 1 at some position.

The same K4 law is evaluated without boolean compression. Correctness is unchanged.

No approximation enters in the regime selection. The three regimes are exact chart specializations of one law.

---

### 10. Common-Source Factorization and Chart Defect

#### 10.1 The chart defect

The chart defect of the K4 lattice matrix is its determinant:

Δ_K4(q, k) = det M(q, k) = D₀₀ D₁₁ − D₀₁ D₁₀

It measures the failure of the four K4 sectors to factor through a single rank-1 source.

#### 10.2 Rank-1 closure for scalars

**Theorem.** For scalar multiplication x · y:

Δ_K4(x, y) = 0

**Proof.** Define chart vectors c_x = (L_x, H_x)ᵀ and c_y = (L_y, H_y)ᵀ. Then M(x, y) = c_x · c_yᵀ, which is rank 1. Therefore det M = 0.

**Corollary.** Scalar multiplication satisfies exact factorization closure: D₀₀ D₁₁ = D₀₁ D₁₀. The four K4 sectors are fully determined by two numbers (the two chart norms). There is no independent information in the cross terms.

This is the arithmetic form of the common-source condition. The product of two scalars factors through a single pair of chart values. All structure traces to one shared decomposition per factor.

#### 10.3 The chart commutator

**Definition.** For a vector v and positions s < t, the chart commutator is:

ω_v(s, t) = L_v[s] · H_v[t] − H_v[s] · L_v[t]

It vanishes when positions s and t have proportional chart decompositions and is nonzero when scale structure varies across the vector.

The chart commutator measures whether two positions in a vector share the same ratio of carrier to gauge content. Proportional charts mean the two positions carry the same scale mixture. Disproportionate charts mean the positions carry independent scale information.

#### 10.4 The Cauchy-Binet decomposition

**Theorem.** For q, k ∈ ℤⁿ:

Δ_K4(q, k) = Σ_{s<t} ω_q(s, t) · ω_k(s, t)

**Proof.** Form the 2 × n chart matrices M_q and M_k with rows (L_v[1], …, L_v[n]) and (H_v[1], …, H_v[n]). Then M(q, k) = M_q · M_kᵀ. By the Cauchy-Binet formula:

det(M_q · M_kᵀ) = Σ_{s<t} det(M_q^{s,t}) · det(M_k^{s,t})

Each 2 × 2 minor determinant is ω_v(s, t).

**Corollary.** The chart defect is the inner product of the two chart-commutator fields. Scalars have zero chart commutators (one position, no pairs). Vectors with inhomogeneous scale structure across positions have nonzero chart defect.

#### 10.5 Connection to the Gram determinant

The K4 lattice matrix M(q, k) = M_q · M_kᵀ, where M_q and M_k are 2 × n chart matrices. Its determinant decomposes via Cauchy-Binet into 2 × 2 minors.

This is structurally parallel to the Gram determinant:

det G(u, v) = ‖u‖² ‖v‖² − ⟨u, v⟩²

In both cases, the determinant measures the failure of two independently placed structures to factor through a single source. Zero determinant means rank-1 factorization (common source). Nonzero determinant means irreducible two-source structure (independent placements carrying independent information).

The Gram determinant operates in inner product space on geometric vectors. The chart defect operates in the dyadic arithmetic chart on integer vectors. Both test the same property: whether two structures share a common measure or require independent measures.

---

### 11. Depth Hierarchy

The chart defect traces the dimensional emergence through the CGM constraint hierarchy.

**Depth 0 (scalar).** M is rank 1. Δ_K4 = 0. The four K4 sectors factor through a single source. All structure satisfies the common-source condition. This corresponds to CS: all structure traces to a common origin.

**Depth 2 (vector).** M becomes rank 2. Nonzero chart commutators create nonzero Δ_K4. Scale inhomogeneity across positions means the L and H charts carry independent information. The order of charts matters: ω_v(s, t) ≠ ω_v(t, s). This corresponds to UNA and ONA: at depth two, the order of operations matters, but the non-commutativity is bounded by the C(n, 2) chart-commutator space.

**Depth 4 (frame closure).** The CGM depth-four balance condition requires b⁴ = id for every byte and XYXY = id for every byte pair. The K4 lattice matrix, as the arithmetic chart of the same transport structure, follows the same affine cancellation over depth-4 frames.

At depth 0, there is one measure (the scalar). At depth 2, the measure has been placed in independent positions (the vector). At depth 4, the independent placements close (balanced frame).

---

### 12. Three-Layer Invariant Structure

The common-source factorization, the finite manifold, and the continuous geometric layer each carry parallel invariant families. These are structurally parallel: each layer has an additive conservation law and a multiplicative closure or defect law. They are not interchangeable: they belong to different layers of the same architecture.

#### 12.1 Arithmetic layer (the K4 chart)

Additive: ρ_carrier + ρ_cross + ρ_align = 1.

Multiplicative: Δ_K4 = det M, vanishing for scalars and decomposing into chart commutators for vectors.

#### 12.2 Finite manifold layer

A discrete realization exists with the following verified properties:

- Reachable manifold |Ω| = 4096 states, with product form Ω = U × V where |U| = |V| = 64
- Dual horizons of 64 states each, satisfying |H|² = |Ω|
- 6-bit chirality register over GF(2)⁶
- 64-point Walsh-Hadamard harmonic basis
- K4 gate group (ℤ/2)² governing spinorial gauge structure
- Self-dual [12, 6, 2] mask code with 64 codewords

Additive: horizon_distance(s) + ab_distance(s) = 12 for every reachable state.

Multiplicative: d_A(s) · d_B(s) = 0.25 for every reachable state. This follows from the pair-diagonal code keeping popcount = 6 per 12-bit component.

#### 12.3 Continuous geometric layer

Additive/closure: Q_G · m_a² = 1/2.

Multiplicative/scale: E_i^UV · E_i^IR = (E_CS · E_EW) / (4π²), the UV-IR product relation.

Aperture: Δ = 1 − δ_BU / m_a.

#### 12.4 Cross-layer structural correspondence (structural correspondence)

The arithmetic 2 x 2 decomposition and the kernel K4 gate group share the (ℤ/2)² organizational pattern across different layers of the architecture.

The following arithmetic-manifold correspondences hold:

| Arithmetic layer | Finite manifold layer | Shared structure |
|-----------------|----------------------|-----------------|
| Rank-1 closure (Δ_K4 = 0) | Common source (CS) | Factorization through shared origin |
| Nonzero chart defect | Non-commutativity (UNA/ONA) | Independent information in distinct placements |
| Depth-4 cancellation | b⁴ = id, XYXY = id | Frame closure |
| K4 cell index {00,01,10,11} | K4 gate group {id,S,C,F} | (ℤ/2)² phase organization |
| Width 64 | |H| = 64, |GF(2)⁶| = 64 | Horizon-matched grain |
| Krawtchouk radial basis | MacWilliams weight transform basis | Shared polynomial engine |

The arithmetic radix satisfies:

```
B = 65536 = |Ω| · 16
```

where the factor 16 is |K4|² = 4². In the continuous geometric layer, 16 = Q_G / (π/4).

The predecessor horizon at k = 5 gives:

```
P₅ = 3 · 2⁴ = 48 = 3 · 16
```

The depth-4 projection of the finite manifold is exactly 4 × 12 = 48 bits. The number 48 combines 16 (the dyadic 4π subdivision factor) with 3 (spatial dimensions). The CGM aperture gap satisfies 48 · Δ ≈ 1 at the frame scale.

At width 64, the chart-commutator space has C(64, 2) = 2016 independent terms. This equals the number of nontrivial swap 2-cycles on Ω:

```
(|Ω| − |H|) / 2 = (4096 − 64) / 2 = 2016
```

The multiplicative defect of the arithmetic chart and the swap-gyration structure of the manifold share the same counted combinatorial skeleton.

#### 12.5 Position in the computational stack

This document develops the arithmetic layer: exact chart decomposition of integer contractions, common-source closure, and defect structure.

The companion document `QuBEC_Climate_Dynamics` develops the operator and transport layer: shell, chirality, gauge, spectral dynamics, and exact partition operator classes.

The aQPU kernel and SDK specifications develop the native state and runtime layer: reachable manifold, transition law, execution surfaces, and observable records.

Practical execution proceeds in this order:

1. state structure (native manifold and charts)
2. operator structure (exact transport and partition algebras)
3. arithmetic lowering (K4 lattice realization and regime selection)

The MacWilliams identity for the self-dual [12,6,2] mask code provides a further cross-layer connection. In coding theory, the MacWilliams transform relates the weight distribution of a code to that of its dual, expressed in the Krawtchouk polynomial basis. In the finite manifold layer, the same Krawtchouk basis diagonalizes shell-radial climate transport. For a self-dual code, the MacWilliams transform becomes a self-consistency condition, and the Plancherel identity on GF(2)^6 guarantees conservation between chirality-space and spectral-space representations. These three structures (MacWilliams weight transform, Krawtchouk radial harmonics, Plancherel spectral conservation) are expressions of a single algebraic duality on the 6-mode register.

---

## Part IV: Geometric Phase and Monodromy

### 13. The Geometric Phase

This section returns to monodromy in geometric-phase language.

#### 13.1 The Berry phase

The geometric phase (Pancharatnam-Berry phase) is the phase difference acquired when a quantum system undergoes a cyclic adiabatic process. Its magnitude equals the solid angle enclosed by the path in parameter space.

For a spin-1/2 particle transported around a closed loop, the Berry phase is half the enclosed solid angle. A complete loop enclosing 4π steradians produces a Berry phase of 2π, returning the system to its original state. A loop enclosing 2π steradians produces a Berry phase of π, flipping the sign of the state (the SU(2) double-cover signature).

#### 13.2 The CGM monodromy as geometric phase

The CGM toroidal holonomy δ_BU = 0.195342 radians is the geometric phase accumulated by the depth-four operational cycle, evaluated in the su(2) representation with canonical stage operators. It measures the angular deficit: the amount by which the system fails to return to its starting state after traversing the full operational loop.

The Foucault pendulum provides a classical illustration. The pendulum swings along one direction (a root process). The Earth rotates beneath it (the angular context). The precession rate is 2π sin φ per sidereal day, where φ is the latitude. The precession is a geometric phase: the memory accumulated when the root process operates within a curved space.

#### 13.3 Branch-point monodromy and Berry phase

The Berry phase around a singular point (a degeneracy, a conical intersection) is structurally identical to the monodromy of a multivalued function around a branch point. In both cases:

- A closed loop in parameter space
- Non-trivial phase accumulation
- The phase is a topological invariant (depends on the enclosed singularity, not the detailed path)

The square-root Riemann surface (Section 7.3) and the Berry phase are two manifestations of the same structure: geometric memory from closed-loop transport around a singular point.

---

### 14. Quadratic Observables and Definite Outcomes (interpretive consequence)

In quantum mechanics, measured quantities are typically quadratic in state amplitudes. The probability of finding a system in state |ψ⟩ upon measurement in basis |φ⟩ is:

P = |⟨φ|ψ⟩|² = ⟨φ|ψ⟩ · conjugate(⟨φ|ψ⟩)

The probability is a square and the amplitude ⟨φ|ψ⟩ is a root. Recovering amplitude from probability requires additional phase information: |α|² alone does not determine the phase of α.

This mirrors the root-square pattern used earlier: quadratic observables preserve magnitude while discarding orientation information. Recovering the full root from the square requires extra structure (for example an interference reference).

The 2.07% aperture, in this reading, is the structural gap that separates the full root information from the recoverable square information. Complete closure (Δ = 0) would mean the square determines the root completely, eliminating the phase ambiguity.

---

## Appendix: Phenomenological Resonances and Research Leads

The following sections document phenomenological resonances and research leads. They play no role in the deductive results of the CGM or the verified properties of the aQPU kernel.

### Appendix P1: Two-Circle Intersection Geometry

#### P1.1 The vesica piscis

The vesica piscis is the shape formed by the intersection of two disks of equal radius, each centered on the perimeter of the other. Its properties include:

- Height-to-width ratio: √3
- Area: (1/6)(4π − 3√3)r² for circles of radius r

The vesica piscis area formula contains CGM geometric invariants: 4π (the quantum gravity invariant Q_G), √3 (from the geometric mean action S_geo), and 1/6 (the reciprocal of the number of edges in K₄).

#### P1.2 The golden ratio and pentagonal scaling

The vesica piscis generates the golden ratio φ = (1 + √5)/2 through a concentric circle construction. The CGM framework identifies the pentagonal scaling:

λ₀ / Δ = 1/√5

where Δ ≈ 0.0207 is the aperture gap. The quantity √5 is the irrational core of the golden ratio.

#### P1.3 The lemon billiard family

The vesica piscis belongs to the lemon billiard family: shapes defined by the intersection of two circles of equal unit radius with centers separated by a distance 2B. The parameter B ranges from 0 (coincident circles) to 1 (tangent circles).

Euclid's geometric construction of √a belongs to this family. Two semicircular arcs intersect with a perpendicular chord, and the root is found at the intersection point. The root is recovered from the two-circle structure through perpendicular (orthogonal) intersection.

---

### Appendix P2: The Lemon Billiard at B = 0.1953

#### P2.1 Context

In the study of quantum chaos, billiard systems serve as fundamental models for investigating the transition between regular and chaotic dynamics. Lozej, Lukman, and Robnik (2022) conducted a systematic survey of approximately 4000 values of B. They identified B = 0.1953 as producing a phase space with specific properties:

- Exactly three major island chains (regular regions)
- One dominant chaotic sea
- No significant stickiness (no partial transport barriers)
- Clean separation between regular and chaotic eigenstates in the semiclassical limit

#### P2.2 Numerical proximity

The CGM dual-pole monodromy defect is δ_BU = 0.195342 radians. The lemon billiard shape parameter producing the uniquely balanced mixed-type phase space is B = 0.1953. Both quantities are dimensionless. Their numerical agreement extends to four significant figures.

#### P2.3 Structural parallels

| Feature | Lemon billiard at B = 0.1953 | CGM at δ_BU = 0.195342 |
|---------|------------------------------|------------------------|
| Geometry | Two-circle intersection | Two modal operators |
| Regular structures | Exactly 3 island chains | 3 rotational DOF (su(2)) |
| Mixed phase space | Regular tori coexist with chaotic sea | 97.93% closure, 2.07% aperture |
| No stickiness | Clean separation, no partial barriers | Clean depth-four closure |
| Semiclassical condensation | Mixed states decay as power law | Aperture is fixed geometric invariant |

The three island chains at B = 0.1953 are particularly notable. Three is the number of independent generators in su(2), the Lie algebra that the CGM framework derives as the unique solution to its foundational constraints.

#### P2.4 Status

The lemon billiard connection is a hypothesis-generating observation. The CGM derives δ_BU from first principles. The quantum chaos community selected B = 0.1953 empirically, through numerical survey, as the parameter producing a uniquely balanced phase space. Whether the agreement reflects a structural identity or a numerical coincidence requires computation of explicit billiard invariants (holonomy, transport flux, or geometric phase) equal to δ_BU, δ_BU/m_a, or Δ.

---

### Appendix P3: Universal Critical Exponent

#### P3.1 The integrability-to-chaos transition

Leonel, de Almeida, Tarigo, Marti, and Oliveira (2026) demonstrate that the transition from integrability to non-integrability in an oval billiard exhibits all hallmarks of a continuous (second-order) dynamical phase transition. The order parameter (the saturation of chaotic diffusion) vanishes continuously as the deformation parameter ε approaches zero, while its susceptibility diverges.

The measured critical exponent is α̃ = 0.507(2), consistent with 1/2. The order parameter scales as:

ω_{rms,sat} ∝ ε^{1/2} = √ε

#### P3.2 Universality

The same critical exponent α = 1/2 appears across multiple distinct systems:

- Oval billiard: α̃ = 0.507(2)
- Fermi-Ulam model: α = 0.5
- Periodically corrugated waveguide: α = 0.5
- Area-preserving maps (γ = 1): α = 1/(1 + γ) = 1/2

The transition from order to chaos universally goes as the square root of the control parameter. The observable that measures the onset of chaos is obtained from the perturbation by root extraction. The half-power exponent (1/2) is the exponent of the root operation itself.

#### P3.3 Connection to the CGM normalization

The CGM normalization:

Q_G · m_a² = 1/2

can be read as:

m_a = √(1 / (2Q_G))

The value 1/2 appears in several independent contexts:

- The SU(2) spin quantum number
- The SU(2) double-cover signature
- The critical exponent governing universal transition from integrability to chaos
- The right-hand side of the CGM normalization condition

#### P3.4 The four questions framework

Leonel et al. propose four questions for investigating dynamical phase transitions. These map to the CGM constraint hierarchy:

| Question | CGM identification |
|----------|-------------------|
| What symmetry is broken? | CS chirality: left-right asymmetry |
| What is the order parameter? | UNA/ONA: degree of non-commutativity |
| What is the elementary excitation? | δ_BU: minimal geometric memory enabling transport |
| Are there topological defects? | Aperture gap Δ: stability islands with measure 2.07% |

#### P3.5 The lemon billiard as open problem

Leonel et al. explicitly identify the lemon billiard as an open problem. The CGM framework suggests a specific prediction: the critical value of the shape parameter is B = δ_BU = 0.195342, determined by the toroidal holonomy of the depth-four closure cycle. Computing the critical exponent and order parameter of the lemon billiard as a function of B would constitute a direct test.

---

### Appendix P4: Biological Reproducibility

The emergence of a dimension through orthogonal placement has the same structural character as biological reproduction. A single genetic system (the common source condition) differentiates into two complementary gametes (orthogonal placements of the same genetic measure), which combine to produce an offspring (the product, the closure) that exists in the next generation (a domain the factors alone do not span).

At the molecular level, DNA replication proceeds by separating a double strand into two complementary single strands (orthogonal differentiation through complementary base pairing), each serving as a template for a new double strand (the product). The two template strands are the same molecule differentiated into two commensurable but distinct placements.

The stomatal aperture in plant leaves provides a structural parallel to the CGM closure-aperture balance. The stoma is a pore formed by two curved guard cells, belonging to the same two-circle intersection family as the vesica piscis and the lemon billiard. The stomatal aperture solves the same optimization: complete closure prevents gas exchange (no observation, no information gain), complete opening causes excessive water loss (no coherent structure), and the optimal aperture balances intake against loss.

Quantitative identification between biological aperture fractions and the CGM value Δ = 0.0207 has not been established. These parallels are structural observations.

---

## Summary of Mathematical Identities

| CGM quantity | Classical identity | Source |
|---|---|---|
| m_a = 1/(2√(2π)) | Derivative of √x at x = 2π | Differential calculus of root function |
| s_p = π/2 | Thales angle for geometric root extraction | Euclid, Elements II.14 and VI.13 |
| Q_G · m_a² = 1/2 | 4π · [f'(2π)]² = 1/2 | Root sensitivity normalization |
| Q_G = 4π | Surface area of quaternionic root sphere S² | Quaternion algebra |
| s_p / m_a² = 4π² | Orthogonality normalized by squared aperture | Optical conjugacy |
| √3/2 in S_geo | Imaginary part of cube roots of unity | Roots of z³ = 1 |
| Σ(roots of unity) = 0 | Sum of nth roots vanishes for n > 1 | Vieta's formulas |
| √n as periodic continued fraction | Periodic non-closure with fixed repeating block | Lagrange (c. 1780) |
| λ₀/Δ = 1/√5 | Golden ratio cut in vesica piscis | Two-circle intersection |
| Area of vesica piscis | (4π − 3√3)/6 · r² | Classical geometry |
| Area(u, v) = ‖u‖ · ‖v‖ under orthogonality | Gram determinant specialization | Bilinear algebra |
| Δ_K4 = 0 for scalars | Rank-1 factorization of K4 matrix | Cauchy-Binet |
| Δ_K4 = Σ ω_q · ω_k for vectors | Chart defect as inner product of commutator fields | Cauchy-Binet |
| Monodromy of √z | One circuit: √z → −√z; two: closure | Riemann surface |
| ‖v‖ = √⟨v,v⟩ | Norm as root of quadratic self-interaction | Hilbert space geometry |
| dim(H ⊗ K) = dim(H) · dim(K) | Dimensions multiply under tensor product | Linear algebra |
| u × v = *(u ∧ v) in 3D | Product of two directions yields third | Hodge duality |
| ω_{rms,sat} ∝ ε^{1/2} | Universal critical exponent for order-chaos transition | Leonel et al. (2026) |
| ρ_carrier + ρ_cross + ρ_align = 1 | Additive sector budget of K4 matrix | Exact arithmetic identity |
| B = |Ω| · 16 | Radix-manifold identity | Cross-layer correspondence |
| C(64, 2) = 2016 = (|Ω| − |H|)/2 | Chart commutator count = swap 2-cycles | Width-64 coincidence |

---

## Open Questions

### On the lemon billiard

Does the lemon billiard at B = δ_BU = 0.195342 exhibit a specific billiard invariant (holonomy, transport flux, geometric phase) numerically equal to δ_BU, ρ, or Δ?

### On the critical exponent

Can the universal critical exponent α = 1/2 for the integrability-to-chaos transition be derived from the CGM normalization Q_G · m_a² = 1/2?

### On the Berry-Robnik parameter

What is the precise Berry-Robnik regular fraction at B = 0.195342 in the lemon billiard? Is it functionally related to ρ = 0.9793?

### On biological aperture

Is the optimal stomatal aperture fraction in plants quantitatively related to Δ = 0.0207?

### On the continued fraction structure

What is the continued fraction expansion of δ_BU? Does its periodic structure relate to the monodromy cycle?

### On the Gram-chart connection

Is there a natural transformation from the Gram determinant of Hilbert space vectors to the chart defect of the K4 lattice matrix that preserves the common-source condition?

### On the combinatorial coincidence at width 64

Is there a structural explanation for the equality C(64, 2) = (|Ω| − |H|)/2 beyond dimensional matching?

---

## References

### CGM Framework

Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. DOI: 10.5281/zenodo.17521384. Repository: github.com/gyrogovernance/science

### Lattice Multiplication History

Chabert, J.-L., ed. (1999). A History of Algorithms: From the Pebble to the Microchip. Springer, pp. 21-26.

Williams, M. R. (1997). A History of Computing Technology, 2nd ed. IEEE Computer Society Press.

Boag, E. (2007). Lattice Multiplication. BSHM Bulletin: Journal of the British Society for the History of Mathematics, 22(3), pp. 182-184.

Nugent, P. (2007). Lattice Multiplication in a Preservice Classroom. Mathematics Teaching in the Middle School, 13(2), pp. 110-113.

Swetz, F. J. (1987). Capitalism and Arithmetic: The New Math of the 15th Century. Open Court.

Smith, D. E. (1968). History of Mathematics, Vol. 2. Dover.

Corlu, M. S. et al. (2010). The Ottoman Palace School Enderun and The Man with Multiple Talents, Matrakci Nasuh. Journal of the Korea Society of Mathematical Education, Series D, 14(1), pp. 19-31.

### Computational Arithmetic

Comba, P. G. (1990). Exponentiation Cryptosystems on the IBM PC. IBM Systems Journal, 29(4), pp. 526-538.

Biham, E. (1997). A Fast New DES Implementation in Software. Fast Software Encryption, LNCS 1267, Springer, pp. 260-272.

Rastegari, M. et al. (2016). XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. ECCV 2016.

### Vedic Mathematics

Nikhilam sutra: https://en.wikibooks.org/wiki/Vedic_Mathematics/Sutras/Nikhilam_Navatashcaramam_Dashatah

### Quantum Chaos and Billiards

Lozej, Č., Lukman, D., and Robnik, M. (2022). Phenomenology of quantum eigenstates in mixed-type systems: lemon billiards with complex phase space structure. arXiv:2207.07197v2.

Heller, E. J. and Tomsovic, S. (1993). Postmodern quantum mechanics. Physics Today 46, 38.

Berry, M. V. and Robnik, M. (1984). Semiclassical level spacings when regular and chaotic orbits coexist. Journal of Physics A: Mathematical and General 17, 2413.

### Dynamical Phase Transitions

Leonel, E. D., de Almeida, M. A. M., Tarigo, J. P., Marti, A. C., and Oliveira, D. F. M. (2026). Describing a Universal Critical Behavior in a transition from order to chaos. arXiv:2602.17810v1.

Leonel, E. D. (2021). Dynamical Phase Transitions in Chaotic Systems. Springer.

### Geometric Phase

Berry, M. V. (1984). Quantal Phase Factors Accompanying Adiabatic Changes. Proceedings of the Royal Society A 392, 45.

Pancharatnam, S. (1956). Generalized Theory of Interference, and Its Applications. Proceedings of the Indian Academy of Sciences A 44, 247.

### KAM Theory

Kolmogorov, A. N. (1954). On the Conservation of Conditionally Periodic Motions under Small Perturbation of the Hamiltonian. Doklady Akademii Nauk SSR 98.

Arnold, V. I. (1963). Proof of a theorem of A. N. Kolmogorov on the preservation of conditionally periodic motions. Uspekhi Matematicheskikh Nauk 18.

Moser, J. (1962). On invariant curves of area-preserving mappings of an annulus. Nachrichten der Akademie der Wissenschaften Göttingen, Math.-Phys. Kl. II, 1.

### Classical Geometry and Algebra

Euclid. Elements, Propositions II.14 and VI.13.

Fletcher, R. (2004). Musings on the Vesica Piscis. Nexus Network Journal 6(2), 95.

### Hilbert Space and Functional Analysis

Reed, M. and Simon, B. (1980). Methods of Modern Mathematical Physics, Vol. I: Functional Analysis. Academic Press.

Hall, B. C. (2015). Lie Groups, Lie Algebras, and Representations (2nd ed.). Springer.

### Physics

Morel, L. et al. (2020). Determination of the fine-structure constant with an accuracy of 81 parts per trillion. Nature 588, 61.

### Gyrogroup Theory

Ungar, A. A. (2008). Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity. World Scientific.

### Gyroscopic ASI Framework

Gyroscopic ASI aQPU Kernel specification and Quantum Computing SDK specification. Repository: github.com/gyrogovernance/superintelligence