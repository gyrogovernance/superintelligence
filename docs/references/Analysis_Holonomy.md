# Analysis: CGM Holonomy

## Path Memory in the Common Governance Model: Continuous Structure and Finite Realization

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**Verification:** `experiments/cgm_holonomy_analysis_run.py` (companions `_common.py`, `_1.py`, `_2.py`). Results are written to `experiments/cgm_holonomy_analysis_results.txt`. Connection classification, the stage-pair precessions, and the Cartesian Thomas Pexp that recovers δ_BU are in [10].

---

## Abstract

Holonomy is the residual transformation that remains when a system is transported around a closed path in a curved space. The traversal returns to its starting point, while the orientation carried along the path does not. This document establishes the holonomy structure of the Common Governance Model (CGM) in three layers. The first layer contains exact algebraic results that follow from the CGM threshold angles alone, including a closed form for the SU(2) commutator holonomy. The second layer places the CGM stages as Einstein speeds in the open gyrovector ball and derives the BU Dual-Pole Loop in closed form as an elementary function of two thresholds, from which the closure ratio ρ ≈ 0.9793 and the aperture gap Δ ≈ 0.0207 follow as definitions. The same angle is recovered from the origin-gyr word, the Ungar gyrotriangle defect on the dual-pole vertices, and the mass-shell geodesic holonomy on the forward hyperboloid. The central structural result of this layer is a conjugation theorem. The palindromic traversal of all payload stages preserves the holonomy angle while transporting its axis, which separates the magnitude of path memory from its orientation. The third layer verifies the finite realization of the same architecture in the Holonomic Quantum Virtual Machine (hQVM), where holonomy appears as an order-two operator structure on a 4096-state manifold.

---

## 1. Scope and Terminology

This document treats one subject, which is the memory that closed paths leave behind in the CGM state geometry. It establishes the definitions, the closed forms, the invariance properties, and the finite realization of that memory. Physical applications of the quantities derived here, including the fine-structure constant and the gravitational coupling, are treated in separate documents [11], [12] and are outside the present scope.

The following vocabulary is used throughout.

A **path** is an ordered sequence of states. A **loop** is a path whose first and last states coincide. Each step of a path contributes a transport operator, and the composition of these operators around a loop is the **holonomy** of that loop. When the holonomy is a three-dimensional rotation, its conjugacy-invariant rotation angle is the **holonomy angle**, and a scalar measurement of nontrivial return is called a **defect**. The holonomy element is characterized by its angle, its axis, its unit quaternion, and its conjugacy class, equivalently by the eigenvalue set {1, exp(+iδ), exp(−iδ)}. The angle is the conjugacy-class invariant. The axis is the oriented realization in a chosen frame.

The production of holonomy follows from the joint role of the four CGM conditions. UNA permits order-dependent composition, so distinct routes to a shared endpoint can differ. ONA keeps those routes mutually comparable within one structure. BU closes the observable configuration while retaining a residual transformation that records which route was taken. Closure of the projected state therefore coexists with a nontrivial transport operator, and that residual is the geometric carrier of path memory.

Two mathematical settings appear. The continuous setting is a **gyrovector space**, which is the algebraic structure formed by relativistic velocity addition inside the open ball of radius c. Velocity addition in this space is neither commutative nor associative, and the correction operator that repairs composition is called the **gyration**. Gyrations are rotations, and they are the source of all continuous holonomy in this analysis. Composition of two Einstein boosts factors as a single boost times a gyration,

```
B(u) B(v) = B(u ⊕ v) Gyr[u, v]
```

so the gyration is the spatial rotational residue of non-collinear boost composition [3]. Palge and Pfeifer identify Thomas–Wigner rotation with holonomy of the Levi-Civita / spin connection on the forward mass shell [4]. The finite setting is the hQVM [5], whose relevant features are introduced in Section 15 before they are used.

---

## 2. The CGM Thresholds

CGM is built from four foundational conditions, named Common Source (CS), Unity Non-Absolute (UNA), Opposition Non-Absolute (ONA), and Balance Universal (BU). Their construction is given in [8]. For the present analysis, each condition contributes one dimensionless threshold, and the analysis depends only on these numbers.

| Condition | Threshold | Value | Character |
|---|---|---|---|
| CS | s_p = π/2 | 1.5707963... | angle |
| UNA | u_p = 1/√2 | 0.7071067... | amplitude, with associated angle arccos(u_p) = π/4 |
| ONA | o_p = π/4 | 0.7853981... | angle |
| BU | m_a = 1/(2√(2π)) | 0.1994711... | amplitude scale |

The UNA threshold is an amplitude whose associated angle is π/4, and the distinction between the amplitude and the angle is maintained throughout.

Write θ_CS = s_p, θ_UNA = arccos(u_p), and θ_ONA = o_p for the three stage angles. At the canonical thresholds these are θ_CS = π/2 and θ_UNA = θ_ONA = π/4. The conditions are nested by logical necessity, and that nesting is realized exactly by the stage angles.

CS fixes the chiral frame at the horizon. UNA is the next necessary condition: without depth-two order contingency, the chiral distinction would have no observable consequence, and the associated angle θ_UNA engages the three rotational degrees of freedom. ONA is the next necessary condition: without non-absolute opposition, the distinctions introduced by UNA would lose recoverable relation to the common source, and θ_ONA engages the three translational degrees of freedom that complete SE(3). BU is not a third lemma. It is the dual of two propositions at depth four: Balance Egress requires commutative closure compatible with depth-two contingency, and Balance Ingress requires that the closed configuration retain enough structure to reconstruct the prior chirality and both non-absolute conditions. Egress and Ingress therefore give access to the preceding stages because those stages are already present as the content that depth-four balance closes and reconstructs.

The nesting appears geometrically as the exact partition of the horizon angle into the two lemma angles,

```
θ_CS = θ_UNA + θ_ONA
```

together with the Euclidean closure of the three stage angles,

```
θ_CS + θ_UNA + θ_ONA = π
```

which is the same identity written in threshold symbols as

```
π/2 + arccos(1/√2) + π/4 = π
```

The right-isosceles partition θ_UNA = θ_ONA = π/4 is forced by the identification of the UNA amplitude with cos(π/4). The second identity links the complete solid angle Q_G = 4π to the BU amplitude scale and yields the half-integer associated with the double cover of the rotation group:

```
Q_G · m_a² = 1/2,   where Q_G = 4π
```

Both identities are consequences of the threshold definitions and of the nested construction of the conditions.

---

## 3. Exact SU(2) Commutator Holonomy

The first holonomy result uses only the threshold angles and the algebra of the group SU(2), the double cover of the rotation group.

Let U be the SU(2) rotation by π/4 about the x axis and let V be the SU(2) rotation by π/4 about the y axis. These are the UNA and ONA stage angles applied about orthogonal axes. The commutator

```
C = U V U† V†
```

measures the failure of the two rotations to commute. For two SU(2) rotations through angles β and γ whose axes have separation δ, the conjugacy angle φ of the commutator satisfies

```
cos(φ/2) = 1 − 2 sin²(δ) sin²(β/2) sin²(γ/2)
```

The threshold configuration sets δ = π/2 and β = γ = π/4. Using sin²(π/8) = (1 − 1/√2)/2,

```
cos(φ/2) = 1 − 2 · ((1 − 1/√2)/2)²
         = 1 − (1 − 1/√2)² / 2
         = (1 + 2√2) / 4
```

and therefore

```
φ_SU2 = 2 · arccos((1 + 2√2) / 4) = 0.5879007626540203 rad = 33.6842°
```

The script computes the commutator with 80-digit matrix arithmetic and confirms the closed form with a residual of 7.4 × 10⁻⁸¹. This angle is the exact continuous benchmark of the analysis. Two rotations whose individual angles are fixed by the CGM thresholds generate, through their commutator alone, a rotation of about 33.7 degrees. Order of operations carries geometric content at these thresholds.

---

## 4. Calibration of the Rotation Machinery

The BU dual-pole and palindrome results below are computed with a software implementation of the gyration operator. Before that implementation is used at the CGM stage coordinates, it is validated against an independent analytic standard.

The standard is the Thomas-Wigner rotation of special relativity. When two boosts with velocities u and v are composed, the result is a boost combined with a spatial rotation, and for small speeds the rotation angle approaches ||u × v|| / (2c²). The calibration evaluates the implemented gyration angle against this formula on a fixed deterministic grid of 576 velocity pairs per speed bound, for maximum speeds from 0.02c to 0.10c.

| max speed (units of c) | fitted slope | max residual |
|---|---|---|
| 0.02 | 1.000132 | 2.0 × 10⁻⁸ |
| 0.03 | 1.000298 | 1.0 × 10⁻⁷ |
| 0.05 | 1.000830 | 8.0 × 10⁻⁷ |
| 0.08 | 1.002128 | 5.2 × 10⁻⁶ |
| 0.10 | 1.003330 | 1.3 × 10⁻⁵ |

The slope error scales with the square of the speed bound (measured order 2.003) and the absolute residual scales with the fourth power (measured order 4.006). Both orders match the known series structure of the Wigner angle, so the implementation reproduces the analytic behavior across the tested range rather than at a single tolerance point. At the CGM stage coordinate magnitudes the matrix layer agrees with the analytic formulae to about 10⁻⁸, and matrix-layer comparisons in this document use that tolerance.

---

## 5. CGM Stage Coordinates in the Gyrovector Space

The dual-pole path departs from the depth-two boundary of the nested lemmas, crosses the Balance Egress and Balance Ingress poles, and returns. In the Einstein gyrovector model with c = 1 the CGM stages occupy the coordinates

```
UNA  = (1/√2, 0, 0)
ONA  = (0, π/4, 0)
BU+  = (0, 0, +m_a)
BU-  = (0, 0, -m_a)
```

Each CGM threshold number is read as an Einstein speed β = ||v|| in the open ball of radius c = 1. The stage coordinates therefore live in the Beltrami–Klein model of the ball, where geodesics are straight chords. Writing γ(β) = 1/√(1 − β²) for the Lorentz factor and η = atanh(β) for the rapidity, the map

```
k(β) = β / (1 + √(1 − β²)) = tanh(η/2)
```

sends each speed to the corresponding Poincaré half-rapidity radius. The orthogonal Wigner closed form of Section 6 is written most directly in these Poincaré radii.

BU appears as a pair of opposite poles on the third axis. That dual is the geometric realization of Balance Universal. The positive pole (BU+) carries Balance Egress, commutative closure derived from UNA. The negative pole (BU−) carries Balance Ingress, memory reconstruction derived from ONA. Because Ingress reconstructs the prior chirality and both non-absolute conditions, the dual-pole structure gives access to the whole nested chain. The second-axis coordinate has magnitude π/4, which is the common lemma angle θ_UNA = θ_ONA. That axis is the depth-two boundary from which the depth-four dual is accessed. The amplitude m_a is the scale of the dual itself. CS supplies the reference frame within which the other stages are defined and is not a location that transport visits. The CS threshold π/2 also exceeds the open unit ball and so cannot serve as a velocity coordinate. All four payload magnitudes lie strictly below 1, so the stage vectors lie in the open ball.

The four stages on the path are the **payload stages**, and CS is the **gauge frame**. The finite machine realizes the same split as an 8-bit instruction with 6 payload bits framed by 2 gauge bits.

---

## 6. The BU Dual-Pole Loop in Closed Form

The central loop departs from the depth-two boundary of the nested lemmas, crosses Balance Egress, crosses Balance Ingress, and returns. In stage coordinates that boundary sits on the second axis at the common lemma angle π/4, so the loop may be written

```
ONA → BU+ → BU- → ONA
```

with the stage label ONA naming that shared depth-two boundary.

This loop is the operational cycle of depth-four balance, the residual phase of the Egress and Ingress cycle. It consists of two gyration corners joined by a pole crossing. The poles of the dual are collinear, so the origin-based gyration gyr(BU+, BU−) equals the identity and the middle edge contributes no rotational residue. The holonomy is therefore generated at the two corners between the depth-two boundary and the dual-pole amplitude. Each corner is a gyration of two boosts, one of magnitude π/4 and one of magnitude m_a, separated by a right angle. The value π/4 is the common lemma angle θ_UNA = θ_ONA of Section 2, so each corner couples the nested lemma content at the depth-two boundary to the dual-pole amplitude m_a. The Wigner angle for boosts of unequal magnitudes β₁ and β₂ separated by an angle θ is Ungar's formula [3]

```
ω(β₁, β₂, θ) = 2 · arctan( sin(θ) k(β₁) k(β₂) / (1 + cos(θ) k(β₁) k(β₂)) )
```

with k as in Section 5. At θ = π/2 one has sin(θ) = 1 and cos(θ) = 0, so the corner angle reduces to

```
ω = 2 · arctan( k(π/4) · k(m_a) )
```

The two corners share the axis parallel to the second-axis boundary times the BU axis. With the signed Ungar convention the generating cross product points along +x and each corner contributes the signed angle −ω, so the loop operator is Rot(+x, −2ω). The identity Rot(+x, −ω) = Rot(−x, +ω) rewrites the same rotation as a positive angle about −x. The unsigned conjugacy angle of the loop, named the **BU Dual-Pole Loop** and written δ_BU, therefore has the closed form

```
δ_BU = 2 · ω = 4 · arctan( k(π/4) · k(m_a) )
```

which is the same quantity written in [9]. This closed form is the definition of δ_BU. Like π, the decimal expansion is infinite; numerical work evaluates the equation rather than substituting a truncated literal. Evaluated at 80-digit precision,

```
k(π/4)  = 0.4851158626411627
k(m_a)  = 0.1007479000361957
ω       = 0.0976710891288310  rad
δ_BU    = 0.1953421782576621  rad  = 11.19°
```

The holonomy therefore consists of a scalar angle together with an oriented axis, and Section 9 shows that these two components behave differently under transport.

---

## 7. The Closure Ratio and the Aperture Gap

Two derived quantities compare the loop defect to the BU amplitude scale.

```
ρ = δ_BU / m_a = 0.9793004544973297

Δ = 1 - ρ      = 0.0206995455026703
```

The **closure ratio** ρ states that the accumulated dual-pole defect fills about 97.93 percent of the aperture scale m_a. The **aperture gap** Δ is the remaining fraction, about 2.07 percent. Within CGM these two numbers carry the balance interpretation developed in [8], where near-closure provides structural stability and the residual gap keeps reconstruction of the system's history possible. In the present document they are definitions. Once δ_BU and m_a are fixed, ρ and Δ contain no further freedom.

The closed form expands in the BU amplitude. With k_ONA = k(π/4) and the series k(m_a) = m_a/2 + m_a³/8 + O(m_a⁵) together with arctan x = x − x³/3 + O(x⁵),

```
δ_BU = 2 k_ONA m_a + (k_ONA/2 − k_ONA³/6) m_a³ + O(m_a⁵)

ρ = 2 k_ONA + (k_ONA/2 − k_ONA³/6) m_a² + O(m_a⁴)
```

so the closure ratio is even in m_a. In the limit of vanishing m_a,

```
ρ(m_a → 0) = 2 · k(π/4) = 0.9702317252823254
```

The baseline gap 1 − 2k(β) is positive precisely when β < 4/5. Because the common lemma angle π/4 is strictly less than 4/5, with margin 4/5 − π/4 = (16 − 5π)/20 ≈ 0.01460, the depth-two boundary already forces a positive baseline gap. The full value exceeds the baseline by a finite-amplitude correction.

```
baseline gap      1 - ρ(0)  = 0.0297682747176746
finite correction ρ - ρ(0)  = 0.0090687292150043
final gap         Δ          = 0.0206995455026703
```

The depth-two boundary at the common lemma angle fixes a closure of 97.02 percent, leaving a baseline gap near 2.98 percent. The finite BU amplitude closes a further 0.91 percentage points, producing the final gap of 2.07 percent. The correction is of second order in m_a with coefficients determined by k(π/4), so the aperture gap is an analytic function of the thresholds with no adjustable content.

---

## 8. Verification of the Closed Form

The closed form is checked at 80-digit precision against independent constructions of the same orthogonal corner and dual-pole loop.

Fix the orthogonal corner with boost speeds β₁ = θ_ONA = π/4 and β₂ = m_a, and let θ = π/2 be their spatial separation. Four routes for the corner gyration agree with δ_BU to residuals at the working floor (~10⁻⁸¹). The first is the raw gyration map: for a probe radius r = 1/2 one forms the columns gyr(r e_i)/r without SVD projection. The second is Ungar's closed SO(3) matrix

```
G = I + α_U Ω + β_U Ω²
```

with scalar coefficients (α_U, β_U) fixed by the boost pair and with Ω the infinitesimal generator of the rotation plane. The third is the spatial 3 × 3 block of the Lorentz factorization L(a ⊕ b)⁻¹ L(a) L(b), where L(·) denotes the Einstein boost in Minkowski space. The fourth is the analytic Wigner angle derived above. Each matrix satisfies SO(3) residuals and Ungar's trace identity at the same floor. Denoting by ε the signed Wigner orientation at θ = π/2, one has ε = −ω and therefore δ_BU = 2|ε| = 2ω. The loop factors as Rot(+x, −ω) Rot(+x, −ω) = Rot(+x, −2ω) because the middle edge is the identity and the two corners share one axis.

Write G_egress, G_middle, and G_ingress for the origin-based gyrations on the successive edges ONA → BU+, BU+ → BU−, and BU− → ONA. The dual-pole **origin-gyr word** is the left-action product

```
R = G_ingress G_middle G_egress
```

The middle factor equals the identity to working precision (collinear poles), the two corner angles agree, and the conjugacy angle of R equals δ_BU. Shared-axis additivity of the two corners is therefore verified as a property of this orthogonal configuration and does not hold for general loop compositions.

---

## 9. The Palindromic Conjugation Result

The BU dual-pole loop uses two of the four payload stages. The full payload traversal visits all of them in a palindromic order.

```
UNA → ONA → BU+ → BU- → ONA → UNA
```

This path places six payload positions on a five-edge closed walk, moving outward from UNA through ONA to the BU pole pair and returning through the same stages in reverse. In the eight-position phase layout of the finite instruction unit the same structure appears as CS | UNA ONA BU | BU ONA UNA | CS, with CS occupying the two outer gauge positions and the six internal positions matching the continuous payload walk. At the central fold the forward and reverse readings of balance join [8]. The computed holonomy angle of this path equals the BU dual-pole angle to working precision, while the rotation axis differs. The BU loop axis is (−1, 0, 0), and the palindrome axis is (−0.9224, 0.3863, 0).

The equality of angles together with the change of axis follows from a theorem of gyrogroup theory [3]. Ungar's inversion identity states that for any two gyrovectors u and v,

```
gyr(v, u) = gyr(u, v)⁻¹
```

so the return leg through a stage pair applies the inverse of the outbound gyration. Let A = gyr(UNA, ONA) be the outbound gyration on the first payload edge, and let H_BU be the dual-pole loop operator of Section 8. Under left action the palindrome operator factors as

```
H_pal = A⁻¹ · H_BU · A
```

Conjugation by A preserves the rotation angle and transports the axis. Writing n_BU and n_pal for the oriented axes of H_BU and H_pal respectively,

```
angle(H_pal) = δ_BU
n_pal = A⁻¹ n_BU
```

The script verifies each component of this statement at 80-digit precision. The reverse gyration gyr(ONA, UNA) matches A⁻¹, the conjugated operator matches the directly composed palindrome, and A⁻¹ n_BU matches n_pal.

The structural content of this result is the separation of path memory into two channels. The magnitude of the memory, the angle δ_BU, is created at the BU pole structure and is invariant under the surrounding traversal. The orientation of the memory, the axis, is transported by the UNA and ONA payload gyration. The outer stages relocate where the memory points without altering how much memory there is.

---

## 10. Gyrotriangle Defect and Stage-Angle Defect

Two defect constructions appear in the continuous layer and must be kept distinct.

Let θ_CS = s_p, θ_UNA = arccos(u_p), and θ_ONA = o_p denote the three stage angles. The **stage-angle defect** is the Euclidean threshold identity

```
δ_stage = π − (θ_CS + θ_UNA + θ_ONA) = 0
```

It closes the angle triangle of the conditions and is not the BU dual-pole aperture.

Let γ(v) = 1/√(1 − ||v||²) be the Einstein factor of a ball vector v. The **Ungar gyrotriangle defect** of a triangle with vertices (a, b, c) is the angular defect δ = π − (α + β + γ) > 0 computed from the three side γ-factors of the Einstein edges. Ungar's gyration–defect theorem identifies the Thomas angle of gyr[u, ⊖v] with the defect of the gyrotriangle generated by u and v [3]. For the corner triangle with vertices at the origin, at ONA, and at BU+,

```
defect(0, ONA, BU+) = ω = δ_BU / 2
```

where ω is the corner Wigner angle, and the same value equals the gyration angle of gyr[ONA, ⊖BU+]. For the closed dual-pole triangle with vertices ONA, BU+, and BU−, gyrotranslation of ONA to the origin yields

```
defect(ONA, BU+, BU−) = δ_BU = 2 · defect(0, ONA, BU+)
```

Both identities hold at the working floor. On a space of constant sectional curvature −1 the angular defect equals the hyperbolic area of the triangle, so Area(ONA, BU+, BU−) = δ_BU in curvature-radius units. Thus the dual-pole aperture is simultaneously an origin-Wigner angle, an origin-gyr word angle, an Ungar triangle defect, and a normalized hyperbolic area on the dual-pole vertices.

Denote Einstein coaddition by u ⊞ v = u ⊕ gyr[u, ⊖v] v. This operation is commutative on the stage vectors. The standard Ungar gyrogroup axioms (gyrocommutativity, left gyroassociativity, even property, gyration inverse, left and right loop properties) hold on the triple (UNA, ONA, BU+) at the same precision.

---

## 11. Mass-Shell Geodesic Holonomy and Distinct Path Objects

Palge and Pfeifer treat Thomas rotation as Levi-Civita / spin-connection holonomy on the forward mass shell V_m⁺ [4]. Here m is the particle mass that fixes the shell radius, while m_a is the dimensionless CGM aperture read as the BU speed. The experiment normalizes to the unit shell m = 1. With stage speeds taken as Einstein betas, each β determines a unit timelike 4-velocity

```
q(β) = (γ(β), γ(β) β)
```

with Minkowski product q · q = +1, so q(β) lies on the forward unit hyperboloid. That hyperboloid carries constant sectional curvature of magnitude 1 (Ricci scalar R = 6 for the positive-definite convention of [4]). Three routes are reported.

**(1) Circular calibration.** Let V ∈ (0, 1) be a fixed spatial speed and write γ(V) for its Lorentz factor. For the constant-speed circle of that speed in the equatorial plane, the curvature integral of the spatial connection yields the circular holonomy angle

```
α_circ(V) = 2π (γ(V) − 1)
```

with residuals at the working floor for V ∈ {0.1, 0.2, 0.3, 0.4, 0.6}.

**(2) Piecewise-geodesic polygon.** Let (q_i) be the successive 4-velocities of a closed stage path. Between q_i and q_{i+1} denote by T_i the unique rotation-free Lorentz boost with T_i q_i = q_{i+1}. The path product

```
P = T_{n−1} ··· T_0
```

is certified in SO⁺(1,3): writing η = diag(+1, −1, −1, −1), one has ||Pᵀ η P − η||, |det(P) − 1|, and the orthochronous residual at the working floor. The SO(3) little-group holonomy at the basepoint is obtained by conjugating P into the rest frame of q_0. Write θ_geodesic for that conjugacy angle. For the BU dual-pole path, θ_geodesic equals δ_BU and equals the origin-gyr word angle at the working floor. The palindrome path yields the same conjugacy angle.

**(3) Chart evaluations of the Palge–Pfeifer connection.** Let ω_s denote the spherical-coordinate pullback of the Palge–Pfeifer spatial connection. Forming the path-ordered exponential P exp(−∫ ω_s) along the same geodesics produces an angle θ_ω-chart ≈ 0.2466 on the dual-pole loop that differs from δ_BU by about 0.051. That offset is a coordinate singularity at the BU poles and at rest. The same connection in Cartesian velocity coordinates, ω = (γ²/(γ+1)) β × dβ, path-ordered with Richardson extrapolation, is regular at rest and at the poles and recovers δ_BU on both the dual-pole loop and the palindrome [10].

A further lab-frame construction is the **relative-boost word**. For successive stage velocities p_i write d_i = ⊖p_i ⊕ p_{i+1} for the Einstein difference and L(d_i) for the corresponding lab-frame boost. The word is the rotational part of the product of the L(d_i). On the dual-pole path its conjugacy angle is about 0.2585 and differs from δ_BU by about 0.063. The operators L(d_i) are lab-frame composites of Einstein differences, whereas the geodesic construction uses the pure transvections T_i, so the two words are different edge operators.

The continuous aperture identity with this speed convention is therefore the agreement of five objects: the closed form δ_BU = 2ω, the origin-gyr word angle of R, the Ungar dual-pole defect, θ_geodesic, and the regular Cartesian Thomas Pexp. The relative-boost word and θ_ω-chart are different constructions.

---

## 12. Dependency Structure

The closed form evaluates the holonomy angle as

```
δ_BU = 4 · arctan(k(π/4) · k(m_a))
```

The factor π/4 is the common lemma angle θ_UNA = θ_ONA. The amplitude m_a is the scale of the dual-pole balance whose poles are UNA-derived Egress and ONA-derived Ingress. The magnitude is therefore the geometric measure of the nested chain at depth four. The shared lemma angle at the depth-two boundary is evaluated against the dual that closes and reconstructs the preceding conditions. CS enters as the gauge frame.

The loop written ONA → BU+ → BU− → ONA is that operational cycle in stage coordinates. ONA labels the depth-two boundary at the common lemma angle. BU+ and BU− are Egress and Ingress.

The sensitivity of the magnitude to the evaluation parameters of the closed form is quantified by logarithmic derivatives at the canonical point.

```
(θ_ONA / δ_BU) · d(δ_BU)/dθ_ONA = 1.61296528
(m_a  / δ_BU) · d(δ_BU)/dm_a    = 1.01888667
```

The response to m_a is close to linear, with the excess above 1 accounted for by the finite-amplitude correction of Section 7. The response to the common lemma angle is superlinear. Finite-difference derivatives of the closed form match the reported logarithmic derivatives at the working difference step.

---

## 13. The Wigner Map at the Canonical Thresholds

The equal-speed Wigner rotation evaluated at the UNA and ONA thresholds, with u_p as the common boost speed and θ_ONA as the separation angle, takes the value

```
ω(u_p, θ_ONA) = 0.2155499101533235
```

At the same point the local geometry of the Wigner map admits closed forms for the partial derivatives

```
dω/dβ = (12√2 - 4) / 17 = 0.7629742793221847
dω/dθ = (21 - 12√2) / 17 = 0.2370257206778153
```

and their sum equals 1 as an algebraic identity. The response of the Wigner angle at the canonical thresholds therefore splits into a boost-magnitude share of 76.3 percent and an angular share of 23.7 percent. The numerical derivatives match the closed forms to better than 10⁻⁴⁰ at working precision.

---

## 14. Precision Governance

The angle δ_BU feeds downstream analyses, including the fine-structure derivation in [11], where the leading expression scales as the fourth power of δ_BU. A relative change ε in δ_BU therefore produces a relative change of about 4ε in that expression.

The closed form above is the definition of δ_BU. The shared evaluator `bu_holonomy_angle` / `BU_HOLONOMY_ANGLE` in `gyroscopic.hQVM.constants` implements that equation. Downstream modules import the shared constant. High-precision work evaluates the same equation at working `mpmath` precision.

---

## 15. The Finite Realization in the hQVM

The Holonomic Quantum Virtual Machine is a finite computational machine [5] that realizes the CGM architecture in exact integer arithmetic. Three of its features are relevant here and are summarized before use.

The instruction unit of the machine is an 8-bit byte whose bit positions carry the four CGM stage labels in palindromic order,

```
CS | UNA ONA BU | BU ONA UNA | CS
```

The two outer bits carry the CS label and act as a frame selector, and the six inner bits carry the payload stages. Reading the four stage labels forward through the first half and comparing with the reverse reading through the second half defines the **fold** of the byte. The four phase pairs are (CS, CS), (UNA, UNA), (ONA, ONA), and (BU, BU). A byte whose two readings agree at every stage position is called flat, and the count k of disagreeing positions, from 0 to 4, measures the byte's internal curvature.

For a fixed set of k disagreeing pairs there are 16 assignments of the common binary values on those pairs, and there are C(4, k) ways to choose which pairs disagree. The number of bytes at disagreement grade k is therefore

```
N(k) = 16 · C(4, k)
```

which produces the distribution

| Disagreeing pairs k | Byte count N(k) |
|---|---|
| 0 | 16 |
| 1 | 64 |
| 2 | 96 |
| 3 | 64 |
| 4 | 16 |

with total 256. The match establishes that the four stage-position comparisons behave as four independent binary observables, each disagreeing in half of all cases. Sixteen bytes are flat and 240 carry curvature. The central BU-to-BU comparison disagrees in 128 of 256 bytes, a fraction of one half.

The machine state lives on a manifold of 4096 states with two six-bit coordinates. The six-bit chirality word χ = u XOR v grades the state between two constitutional horizon sectors. Certain distinguished instruction words of length two, named W2 and W2', act on this manifold as involutions, meaning operators that square to the identity. In the six-bit chart, W2 flips all six chirality bits by the mask 63 = 2⁶ − 1, exchanging a chirality word with its complement, and maps the shell grade s to 6 − s, thereby exchanging the two extremal regions of the state space. Together with the identity and their product they form a Klein four-group, a commutative group of four elements each of order at most two, referred to as K4.

The continuous dual poles BU+ and BU− are realized finitely by this W2 pole exchange [5], [6]. Balance Egress is the involution property that W2 squared equals the identity on the full manifold. Balance Ingress is the invertible pairing of each state with a unique shadow under W2, so that a second application reconstructs the original [5], [7]. Egress and Ingress are simultaneous readings of one depth-four operator.

The conjugate half-word W2' is the second depth-four factor in the Klein four-group. Their operator product

```
F = W2 ∘ W2'
```

yields the Z2 carrier sheet flip while preserving shell. The product F squared returns the carrier to rest and completes one Z2 holonomy cycle [5], [6].

The canonical W2 and W2' certificate passes in full. The operators are involutions, they exchange the extremal state regions as required, and the associated shell and chirality transformations hold on all 4096 states. These finite results carry no numerical tolerance because the arithmetic is exact. The full K4 operator-product table and its permutation spectrum are developed in [5].

---

## 16. Byte-Horizon Aperture Quantization

The continuous aperture gap Δ is fixed above from the holonomy ratio. The instruction unit is an 8-bit byte, so the natural discrete scale at that horizon is 256 ticks. The byte-horizon quantization of the aperture is the nearest integer number of ticks to 256 · Δ.

```
256 · Δ = 5.299083648683975

round(256 · Δ) = 5

Q_256(Δ) = 5/256 = 0.01953125
```

Thus the finite aperture at the byte horizon is five ticks open out of 256. Relative to the continuous gap,

```
|Δ − 5/256| / Δ = 0.05645
```

so the dyadic value lies about 5.6 percent below Δ. This is the quantity written A_kernel = 5/256 in the measurement and QuBEC reports, and stored in the shared constants module as APERTURE_GAP_Q256 = 5.

The same gap participates in the depth-four closure count of the machine. Four successive bytes contribute 48 chirality bits (4 · 12), and

```
48 · Δ = 0.9935781841281744
```

which sits 0.64 percent below unity. The reciprocal scale 1/48 is therefore the depth-four companion of Δ, while 5/256 is its expression at the single-byte horizon. The turn-normalized holonomy δ_BU / (2π) ≈ 1/32 supplies the third natural scale; the ratio (1/48) / (1/32) = 2/3 is the chirality-to-space factor developed in the QuBEC theory analysis. In the present document the operative finite statement is the byte-horizon identity

```
Q_256(Δ) = 5/256
```

with Δ as above.

---

## 17. Continuous-Finite Correspondence

The continuous and finite layers realize the same architecture at different formal levels. The following table records the correspondence of structural roles.

| Continuous layer | Finite hQVM layer |
|---|---|
| closed path in the gyrovector space | operator word on the 4096-state manifold |
| holonomy angle as conjugacy invariant | nontrivial finite involution |
| BU dual-pole loop | W2 pole exchange |
| Balance Egress and Balance Ingress | simultaneous readings of W2 |
| closure under return | W2 squared equals the identity |
| invertible memory of the prior state | W2 pole pairing |
| operator product of the two depth-four half-words | F = W2 ∘ W2′, Z2 sheet flip |
| full return to rest | F squared, Z2 holonomy cycle |
| aperture gap Δ | byte-horizon dyadic 5/256 |
| palindromic payload path | byte fold across the central BU boundary |
| 6 payload positions | 6 payload bits |
| CS gauge frame | byte bits 0 and 7 as frame selector |
| conjugacy spectrum 1, exp(±iδ) of the SO(3) rotation | involution spectrum +1, -1 |


The continuous BU holonomy carries a general rotational phase with eigenvalues 1 and exp(±i δ_BU). The finite carrier operators, under their permutation-matrix lift, carry an order-two phase distinction with eigenvalues in {+1, −1}.

The layers localize curvature at related but distinct sites. In the continuous realization the holonomy is generated at the two corners between the depth-two boundary and the dual-pole amplitude, while the pole crossing is flat. In the finite realization the fold disagreement is counted at the central BU boundary of the byte. Both descriptions organize a dual return through the balance stage.

The six payload positions of the continuous palindrome and the six payload bits of the byte both count degrees of freedom within the three-dimensional, six-degree-of-freedom framework that CGM derives. They are combinatorial structures inside that framework rather than additional spatial dimensions.

---

## 18. Falsification Criteria

The analysis fails if any of the following occurs.

1. The 80-digit SU(2) matrix computation departs from the closed form 2 · arccos((1 + 2√2)/4).
2. An independent realization of the stage speeds as Einstein betas yields a value that differs from δ_BU = 4 · arctan(k(π/4) · k(m_a)) beyond its stated numerical floor.
3. The origin-gyr word, the Ungar defect on (ONA, BU+, BU−), or the mass-shell geodesic holonomy departs from δ_BU beyond the working floor.
4. The palindrome holonomy departs from the conjugacy class of the BU Dual-Pole Loop, in angle or in transported axis.
5. The byte fold distribution departs from 16 · C(4, k), or the W2 and W2' certificate fails on any of the 4096 states.
6. The nearest 8-bit dyadic to Δ departs from 5/256, or the shared constant APERTURE_GAP_Q256 departs from 5.

---

## 19. Reproducibility

```
python experiments/cgm_holonomy_analysis_run.py
python experiments/hqvm_wavefunction_kernel.py --k4-only
```

The first command runs all integrity checks, prints the full report, and writes `experiments/cgm_holonomy_analysis_results.txt`. The run is deterministic, uses 80-digit arithmetic for the analytic layer, and exits with a nonzero code if any check fails. Companions are `cgm_holonomy_analysis_common.py`, `_1.py`, and `_2.py`. The second command reproduces the finite certificate independently.

---

## 20. Conclusion

Three results form the foundation established here. The CGM threshold angles generate a nontrivial SU(2) commutator holonomy with the exact closed form 2 · arccos((1 + 2√2)/4). On the dual-pole path with stage thresholds as Einstein speeds the holonomy angle is δ_BU = 2 · ω = 4 · arctan(k(π/4) · k(m_a)), an elementary function of the common lemma angle and the dual-pole amplitude, from which the closure ratio of 97.93 percent and the aperture gap of 2.07 percent follow as definitions, with the gap decomposing into a 2.98 percent baseline fixed by the depth-two boundary and a 0.91 percentage point closure supplied by the finite BU amplitude. The same angle is the origin-gyr word, the Ungar dual-pole gyrotriangle defect, the mass-shell geodesic holonomy, and the regular Cartesian Thomas Pexp; the relative-boost word and spherical-chart Pexp are different constructions. The palindromic traversal of the payload stages conjugates this holonomy, preserving the angle while transporting the axis, so the magnitude of path memory is set at the balance stage and its orientation is steered by the surrounding stages.

The finite machine realizes the same architecture in exact arithmetic. Byte-level fold curvature is distributed binomially, and the balance-stage exchange operators are involutions on the full state manifold [5]. At the byte horizon the continuous aperture quantizes as Q_256(Δ) = 5/256. The quantities established here, in particular δ_BU, ρ, Δ, and the dyadic aperture 5/256, are the fixed inputs that downstream analyses of physical couplings and the hQVM kernel consume.

---

## References

[1] A. A. Ungar, Beyond the Einstein Addition Law and Its Gyroscopic Thomas Precession, Springer (Kluwer), Dordrecht (2001).

[2] A. A. Ungar, Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity, 2nd ed., World Scientific, Singapore (2008).

[3] A. A. Ungar, Gyrations: The Missing Link Between Classical Mechanics with its Underlying Euclidean Geometry and Relativistic Mechanics with its Underlying Hyperbolic Geometry, arXiv:1302.5678 (2013).

[4] V. Palge, C. Pfeifer, Thomas–Wigner rotation as a holonomy for spin-1/2 particles, Physical Review A 109, 032206 (2024), arXiv:2310.08121.

[5] Analysis_hQVM_Wavefunction.md, docs/Findings. Finite K4 structure, W2 pole exchange, and Balance Egress and Ingress as simultaneous readings of W2 (Theorems T1 through T10).

[6] Analysis_Gravity_Note.md, docs/Findings. Depth-four half-word W2 as involutive Egress, operator product F = W2 ∘ W2′ as the two-pass holonomy cycle, and spin-2 factor from that return.

[7] Analysis_hQVM_CGM_Group_Theory.md, docs/Findings. Finite group presentation of balance. The same W2 supplies involutive closure and invertible pole pairing.

[8] CGM_Logic.md, docs. Construction of the conditions as a chain of necessities. Dual balance, palindromy, and the fold where forward and reverse readings of balance join.

[9] Analysis_CGM_Constants.md, docs/Findings. Stage thresholds and the dual-pole loop definition of δ_BU.

[10] Analysis_Precession.md, docs/Findings. Connection classification, stage-pair precessions, and the Cartesian Thomas Pexp that recovers δ_BU.

[11] Analysis_Fine_Structure.md, docs/Findings. Downstream use of δ_BU in the electromagnetic coupling.

[12] Analysis_Gravity.md, docs/Findings. Downstream use of the aperture structure in the gravitational coupling.
