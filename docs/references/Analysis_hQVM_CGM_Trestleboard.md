# Analysis: hQVM CGM Trestleboard

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**Reproducibility:** `experiments/hqvm_cgm_trestleboard_results.txt`. Scripts: `hqvm_cgm_trestleboard_1.py` through `hqvm_cgm_trestleboard_5.py`, `hqvm_cgm_trestleboard_common.py`, `hqvm_cgm_trestleboard_run.py`. External data catalogs: `data/catalogs/ensdf/` and `data/catalogs/fusion/` (SOURCE files in each directory). Local PDF copies of primary isomer references: `docs/references/` (`SOURCE_Th229m.txt`, `SOURCE_U235m.txt`).

**Subject classes (arXiv-style):** nucl-th; nucl-ex; physics.plasm-ph; math-ph

**Keywords:** Common Governance Model, hQVM trestleboard, nuclear isomer, Th-229m, Delta-ruler, fusion S-factor, percolation hierarchy, alpha decay, beta decay, Coulomb barrier, nuclear magic numbers

## 1. Scope and Claims

### 1.1 Scope

This document demonstrates that a discrete geometric framework predicts specific nuclear and fusion observables with high precision and zero free nuclear parameters. The framework derives a single logarithmic energy coordinate from the electroweak scale. This coordinate places electroweak masses, nuclear binding energies, and isomeric excitations on the same ruler. The primary empirical results are:

*   The Th-229m isomeric excitation energy is predicted to be 8.3563 eV. The measured value is 8.3557 eV (relative error 7.19e-05).
*   The deuteron binding energy is predicted to be 2.2242 MeV. The measured value is 2.2240 MeV (relative error 8.89e-05).
*   The ruler unit that governs these nuclear scales is derived from the W and Z boson mass ratio. Two independent derivations yield this unit with an absolute agreement of 8.34e-10.
*   An algebraic mapping of nuclear quantum numbers preserves chirality shell and parity across 314/314 alpha parents and 801/801 beta parents in the IAEA LiveChart catalog.
*   The seven canonical nuclear magic numbers 2, 8, 20, 28, 50, 82, and 126 emerge as large-gap closures in a Nilsson spectrum whose spin-orbit and deformation couplings are fixed by the same geometric constants chain that sets the electroweak ruler.
*   The Coulomb barriers for seven fusion fuels map to a single structural rung on this ruler. Known resonances for five of these fuels align with structural percolation thresholds.

These results follow from the Common Governance Model (CGM), a Hilbert-style axiomatization of fundamental physics and information science. CGM begins from a single foundational principle about ancestry preservation and develops subsequent structure through a four-stage sequence of recursive operations. Within that construction, the Holonomic Quantum Virtual Machine (hQVM) is the discrete realization of CGM on a finite 4096-state register. This executable kernel supplies exact combinatorial, spectral, and percolation data with no freely adjustable parameters.

This analysis introduces three procedures to map physical observables onto the kernel. Following the nomenclature of architectural drafting, we refer to these procedures as the Level, the Square, and the Compass. The Level locates an absolute energy on the shared logarithmic ruler and identifies the nearest grammar class. The Square assigns that placement to an energy sector and returns the percolation coverage for fusion contexts. The Compass traces the transition between two energies as an explicit sequence of moves. The entire analytical workspace, which unifies these tools on a single coordinate system, is termed the trestleboard.

The numerical claims rest on a small set of fixed energy placements. Optional landmarks serve only as scaffolding.

### 1.2 External Data

Empirical verification uses published nuclear structure data (ENSDF / IAEA LiveChart), the measured Th-229m isomer energy of Zhang et al., PDG electroweak masses and the deuteron binding energy, reference astrophysical S-factor fits for fusion holdouts, and the Rider terrestrial fusion viability boundary. Catalog paths, filenames, and full provenance are collected in Appendix B.

### 1.3 Prior Documents Used Here

This analysis draws on four companion documents in the CGM corpus. Each is cited by filename when its content is used, and the quantities needed for the nuclear and fusion claims are defined in this document at the point of use rather than assumed known.

`Analysis_CGM_Constants.md` supplies the geometric constants that fix the energy spacing of the shared ruler. `Analysis_Compact_Geometry.md` supplies the finite algebraic kernel and the spectral expansion that recovers the electroweak masses. `Analysis_hQVM_Percolation.md` supplies the coverage hierarchy used by the Square and by the fusion resonance map. `docs/CGM_Logic.md` records the minimum-necessity chain of which the nuclear isomer placement is a member.

## 2. The Three Instruments and the Delta-Ruler

The Square, Compass, and Level share one energy coordinate. That coordinate is fixed first, then each tool is defined on it.

### 2.1 The Aperture Gap and the Delta-Ruler

Nuclear binding energies near a few MeV, isomeric excitations near a few eV, and electroweak masses near the hundred-GeV scale all receive a coordinate on that shared logarithmic ruler. A shared coordinate of this kind is usable only if its spacing unit is fixed once, independently of which energy band is under study. This section constructs that unit from two geometric inputs of the CGM kernel. Two named quantities must be kept distinct: the aperture m_a is a fixed reference scale, and the aperture gap Δ is the fractional opening relative to that scale. Only Δ becomes the ruler tick.

The first input is the aperture

```
m_a = 1 / (2 √(2π)) ≈ 0.199471
```

m_a is a dimensionless closed-form constant. Geometrically, it is the largest residual oscillation about the closed configuration that remains inside a total phase of π, half of a full 2π cycle (`Analysis_CGM_Constants.md`). In this document that constant is the reference scale against which the monodromy is compared.

The second input is the dual-pole monodromy

```
δ_BU ≈ 0.195342 rad
```

δ_BU is a geometric phase angle, in radians, accumulated by traversing a closed loop on the kernel. It is recovered from the SU(2) half-loop trace (`Analysis_Monodromy.md`). Numerically δ_BU lies slightly below m_a.

The closure ratio and the complementary opening are

```
ρ = δ_BU / m_a ≈ 0.979300
Δ = 1 − ρ = 1 − δ_BU / m_a ≈ 0.020699
```

ρ is the closure ratio, the fraction of the aperture filled by the monodromy (about 97.93 percent). Δ is the aperture gap, the complementary open fraction (about 2.07 percent). Δ is dimensionless. It is the spacing unit of the energy ruler used for the rest of this document.

The ruler coordinate for an energy E is defined by the pair of maps

```
n(E) = log2(v / E) / Δ ,   E(n) = v / 2^(n Δ)
```

Here E is an absolute energy and v = 246.22 GeV is the electroweak vacuum expectation value placed at the origin n = 0. The quantity n(E) is the tick coordinate of that energy on the Δ-spaced logarithmic ruler. Energies above v give negative n. Energies below v give positive n. One tick is one unit of n. The residual of a measured energy against a named grammar class is n(E) − n(class).

Three conversions on this ruler recur throughout the trestleboard:

```
ticks per aperture grade  = log2(1/Δ) / Δ = 270.26
ticks per ρ-layer         = log2(1/ρ) / Δ = 1.458
ticks per octave          = 1 / Δ         = 48.31
```

These are the tick widths of one aperture-grade step, one holonomy dress layer, and one factor-of-two energy change, respectively. The equatorial code factor 2^(C3 Δ²) = 1.00595755, with enumerator weight C3 = 20, appears later when a grammar class sits on the nuclear equator.

The same ruler carries electroweak masses at small n, nuclear optical structure at large n, and atomic levels at the largest n. Section 2.2 recovers the same Δ independently from the W/Z mass ratio, so the nuclear placements that follow do not rest on an unchecked geometric input.

### 2.2 Independent Recovery of Δ from the W/Z Mass Ratio

The tick distance between the W and Z masses on the ruler is `n_W − n_Z = log2(m_Z / m_W) ≈ 0.182`, with `m_Z` and `m_W` the measured boson masses and `n_W`, `n_Z` their tick coordinates. Before any nuclear claim is made, the aperture gap `Δ`, already fixed by the constants chain, is recovered independently by Newton inversion of the electroweak mass-coordinate expansion of `Analysis_Compact_Geometry.md`. Each channel mass is a carrier-trace polynomial

```
L_i = a_i Δ + b_i + c_i Δ² + p_i Δ³/√5 + q_i Δ⁴ + r5_i Δ⁵
```

The index `i` runs over the four channels Top, Higgs, Z, and W. The coefficients `a_i, b_i, c_i, p_i, q_i, r5_i` are kernel constants drawn from the horizon size `|H| = 64`, the enumerator weights `C1 = 6`, `C2 = 15`, `C3 = 20`, the reduced shell moment `M_shell = 192`, the K4 stage flags, and the trace-free edge increments. At fifth order the four channels recover their PDG masses with tick error below `6.15e-9`. The value of `Δ` that recovers the measured `m_W / m_Z` agrees with `Δ_ref` to absolute error `8.34e-10`. That `Δ_ref` is the ruler unit throughout this document.

With the PDG ratio `m_Z / m_W = 1.134470`,

```
log2(m_Z / m_W) = 0.182019026 = n_W − n_Z
```

the W/Z code gap `C2 − C1 = 9` enters the D4 kernel identity

```
log2(m_Z / m_W) = Δ · S_WZ(Δ)
S_WZ(Δ) = (C2 − C1) − (C3/2)·Δ + 2·Δ²/√5 − Δ³
```

in which `S_WZ` is a dimensionless polynomial in `Δ` whose coefficients are the same kernel constants. Newton inversion of that identity from the measured mass ratio yields a second determination of the same aperture:

```
Δ_WZ = 0.020699554747
Δ_ref = 0.020699553913
|Δ_WZ − Δ_ref| = 8.340e-10
```

with relative error `4.029e-08`. The two determinations, one from the full four-channel `L_i` expansion and one from the W/Z ratio identity alone, agree to the fourth-order D4 target of the compact-geometry analysis (absolute error `8.34e-10`). All nuclear and fusion placements in this document use that `Δ`.

### 2.3 Closure Grammar

A grammar class is a pair of non-negative integers `(k, ℓ)`. The first integer `k` counts aperture grades `Δ^k` below the anchor. The second integer `ℓ` counts the holonomy dress `ρ^ℓ`. Dress meanings used in the trestleboard are

```
ℓ = 0   bare (no holonomy)
ℓ = 2   Z2 two-pass spinorial (F = W2 o W2')
ℓ = 4   EM-depth dress (dual commutator scale)
ℓ = 5   STF gravity bulk (five shells, rho^5)
```

The class predicts an energy through

```
E(k, ℓ) = v · Δ^k · ρ^ℓ · (1/√5 if STF) · (2^(C3 Δ²) if equatorial tick)
```

The optional factors are the STF equipartition `1/√5`, applied when the class carries trace-free quadrupole dress, and the equatorial tick `2^(C3 Δ²)`, applied when the class sits on the nuclear/spinorial equator. Both flags are fixed grammar properties of the class. A sector is a coarser band of `n` used only for narrative grouping and tables (Section 2.5). Grammar classes remain the pairs `(k, ℓ)`. Two classes are forced by the grammar. The remaining classes are optional boundary landmarks used by the Compass and Square.

```
status    (k,ℓ)   n          E              role
FORCED    (3,0)   810.78     2.1838 MeV     Strong bare (v·Δ³)
FORCED    (6,2)  1680.15     8.3563 eV      Nuclear spinorial (v·ρ²·Δ⁶/√5·2^(C3Δ²))
optional  (6,0)  1621.56    19.368 eV       Boundary bare (v·Δ⁶)
optional  (6,4)  1683.48     7.9665 eV      Boundary EM (v·ρ⁴·Δ⁶/√5)
optional  (6,5)  1684.93     7.8016 eV      Boundary gravity (v·ρ⁵·Δ⁶/√5)
optional  (4,0)  1081.04    45.203 keV      keV bare (v·Δ⁴)
optional  (4,2)  1140.04    19.387 keV      keV spinorial (v·ρ²·Δ⁴/√5)
optional  (3,2)   869.78   936.60 keV       Strong spinorial (v·ρ²·Δ³/√5)
optional  (3,5)   874.15   879.63 keV       Strong gravity (v·ρ⁵·Δ³/√5)
```

### 2.4 The Level

The Level places an absolute energy on the ruler. From an energy `E` it returns `n(E)`, the named grammar class whose predicted energy `E(k,ℓ)` minimizes the absolute tick residual `|n(E) − n(E(k,ℓ))|`, and that residual. Against the forced classes,

```
Th-229m     n = 1680.15   class (6,2)   tick = +0.00501   |rel| = 7.189e-05
Deuteron BE n =  809.51   class (3,0)   tick = -1.27239   |rel| bare = 1.809e-02
```

The deuteron full formula `v·Δ³ + v·Δ⁴·(2/√5)` closes at `|rel| = 8.891e-05` against the measured 2.2240 MeV. The 1.272-tick Level residual is a code-atom resolution limit of the underlying kernel lattice.

### 2.5 The Square and Sector Placement

The Square reads the percolation hierarchy and a coarse sector band of the energy ladder. From an energy `E` it reports `n(E)` and the sector band. When a fusion barrier is also supplied, it reports the inclusion probability `p` and the coverage `θ(p)`. The sector is only a narrative band of `n`. Grammar classes remain the pairs `(k, ℓ)`. The sector thresholds are

```
n < 0        Planck / CS
0 ≤ n < 200  EW / UV
200 ≤ n < 900  Strong / IR
900 ≤ n < 1200 keV / Plasma
1200 ≤ n < 1900 Nuclear / Boundary
n ≥ 1900     Atomic / Deep IR
```

Measured placements on the shared ruler are

```
object          n         sector
EW v              0.00    EW/UV
Z                69.23    EW/UV
W                78.02    EW/UV
Deuteron        809.51    Strong/IR
10 keV plasma  1186.18    keV/Plasma
Th-229m        1680.15    Nuclear/Boundary
Cs hyperfine   2537.48    Atomic/Deep IR
```

In the fusion module the Square becomes the coverage dial. The inclusion probability is `p = E / V_b`, and the coverage `θ(p)` sets a resonance-independent lower bound on the fusion rate.

### 2.6 Optical Conjugacy and the Horizon Lemma

Optical conjugacy on the ruler pairs an ultraviolet energy with an infrared conjugate through the kernel constant

```
K = E_CS · v · (1/(4π²)) ,   E_conj = K / E ,   OPTICAL_DILUTION = 1/(4π²)
```

Taking `E_CS = 1.22 × 10^28` eV as the Planck-scale reference and `v` as the electroweak anchor in eV, the product identity `E · E_conj = K` follows at once, and the tick sum `n_UV + n_IR = −log2(K/v²)/Δ` is constant for every stage. For the electroweak anchors the residuals vanish to machine precision:

```
object   E·E_conj resid   n_sum resid
EW v     0.00e+00         0.000e+00
Z        0.00e+00         0.000e+00
W        0.00e+00         0.000e+00
```

Sample conjugates are `EW v → 3.090296e+26 eV` at `n = -2423.08` and `Z → 8.344256e+26 eV` at `n = -2492.31`.

The Horizon Lemma places dyadic and predecessor horizons on the `2^a · 3^b` table (`hQVM_Specs_Formalism.md`). Verified on-table values include

```
n      factorization   role
6      2^1 · 3^1       predecessor
9      2^0 · 3^2       2^a 3^b
24     2^3 · 3^1       predecessor P_k
64     2^6 · 3^0       dyadic (|H|)
192    2^6 · 3^1       predecessor
1536   2^9 · 3^1       predecessor
4096   2^12 · 3^0      dyadic (|Ω|)
```

Octave moves on the Compass are Horizon-Lemma dyadic jumps. An energy halving `E → E/2`, or a doubling `E → 2E`, shifts `n` by one octave, equal to `1/Δ = 48.31` ticks.

### 2.7 The Compass and Explicit Paths

The Compass traces the holonomy dress between two energies. From a start energy and an end energy it returns the ordered list of moves that carry the start anchor class to the end anchor class, together with any residual tick. Anchor selection keeps the Level class when the absolute tick residual is below half an aperture grade (`|n − n_cls| < ½ · 270.26` ticks), and otherwise snaps to the nearest bare grade `k = round(n / 270.26)`. Dress ranks are restricted to the ordered ladder `DRESS_ORDER = (0, 2, 4, 5)`.

Five move types appear, and the Compass applies them in a fixed routing order:

1. Undress along `DRESS_ORDER` until `ℓ = 0` (remove holonomy layers `rho^(-1)`).
2. Δ-step along bare grades until `k` matches the target (`Δ^(±1)`).
3. Dress along `DRESS_ORDER` until `ℓ` matches the target (`rho^(+1)`).
4. Octave if the residual tick gap equals one octave (`1/Δ ≈ 48.31` ticks) within tolerance (`E → E/2` or `E → 2E`).
5. Code or offset: the residual is snapped to the nearest named code atom in `{C1, C2, C3, halves, differences, sums}`, or recorded to the measured energy when no code atom fits.

Compass offsets are measurement residuals relative to the discrete code-atom set. They leave `Δ`, `v`, `ρ`, and the other upstream constants unchanged.

Each dress move cites its operator. The two-pass spinorial closure is `F = W2 o W2'`, with `W2 = (0xaa, 0xab)` and `W2' = (0x2a, 0x2b)`. The word lifts the 24-bit carrier to the 32-bit spinorial frame and preserves the chirality shell. The involution `F^2 = id` holds on all 64 micro-refs.

Five measured Compass paths connect the nuclear and fusion scales.

The path from 10 keV to the deuteron starts at the `(4,2)` keV spinorial class and ends at the `(3,0)` strong bare class:

```
1. undress  ρ^-1   ℓ=2→0    Δticks=59.00   E 19.39 keV → 45.20 keV    F^-1 = F
2. Δ-step   Δ^-1   k=4→3    Δticks=270.26  E 45.20 keV → 2.184 MeV   aperture
3. offset   -1.272 ticks    Δticks=1.27    E 2.184 MeV → 2.224 MeV   bound-state residual
```

The path from the deuteron to Th-229m starts at `(3,0)` strong bare and ends at `(6,2)` nuclear spinorial with no residual remaining:

```
1. Δ-step   Δ^+1   k=3→4    Δticks=270.26  E 2.184 MeV → 45.20 keV
2. Δ-step   Δ^+1   k=4→5    Δticks=270.26  E 45.20 keV → 935.7 eV
3. Δ-step   Δ^+1   k=5→6    Δticks=270.26  E 935.7 eV → 19.37 eV
4. dress    ρ^+2   ℓ=0→2    Δticks=58.59   E 19.37 eV → 8.356 eV     F = W2∘W2′
```

The path from EW to the deuteron takes three aperture steps from the anchor to `(3,0)` and then the same 1.272-tick bound-state residual. The path from the barrier (about 0.44 MeV) to the deuteron starts at `(3,5)` strong gravity, applies three undress layers (`ℓ = 5 → 4 → 2 → 0`) to return to the bare strong scale, and then applies the deuteron residual. The longest measured path, from 10 keV to the barrier, undresses from `(4,2)` to bare keV, takes one aperture step to strong bare, dresses through `ℓ = 2, 4, 5`, applies one octave `E → E/2`, and finishes with a 0.660-tick residual to the barrier energy.

```
1. undress  ρ^-1   ℓ=2→0    Δticks=59.00
2. Δ-step   Δ^-1   k=4→3    Δticks=270.26
3. dress    ρ^+2   ℓ=0→2    Δticks=59.00
4. dress    ρ^+2   ℓ=2→4    Δticks=2.92
5. dress    ρ^+1   ℓ=4→5    Δticks=1.46
6. octave   E→E/2           Δticks=48.31
7. offset   -0.660 ticks    Δticks=0.66
```

Self-checks confirm that dress ranks stay in `{0, 2, 4, 5}`, that the Deuteron→Th path has no residual offset, that the 10 keV→Deuteron path carries the bound-state residual, and that the 10 keV→Barrier path uses an octave.

### 2.8 The Three Instruments as One Geometry

The Square, Compass, and Level are three readings of one object, the CGM energy grammar on the finite kernel. The Level gives the absolute coordinate. The Square gives the channel-accessibility structure at that coordinate. The Compass gives the move sequence that the kernel executes between coordinates. A fusion or decay prediction in this document states where on the ruler a transition lands and which percolation event opens there.

## 3. Kernel Percolation Foundation

### 3.1 The Reachable Manifold and the Square-Root Cluster Theorem

The kernel reachable set `Ω` contains 4096 states. Ancestry preservation forces `Ω` to factorize as a product of two conjugate faces `U` and `V`, each of size 64, so `|Ω| = |H|^2 = 4096` with `|H| = 64` the constitutional horizon. Under a fiber-complete restriction, which preserves the bipartite carrier factorization into the faces `U` and `V`, the reachable cluster from rest satisfies the square-root cluster theorem:

```
|Reach(A)| = root(A)^2 = (2^r(A))^2
```

Here `r(A)` is the GF(2) transport rank of the allowed byte set `A`, and `root(A) = 2^r(A)` is the surviving root dimension. In log2 coordinates this is the linear identity `log2|Reach(A)| = 2 r(A)`, slope 2 fixed by the product geometry. The identity holds at every transport rank under fiber-complete restriction and across the hQVM(`d_χ`) kernel family, where chirality dimension `d_χ` generalizes the physical instance `d_χ = 6` studied here.

The shell census verifies the factorization. With shell index `s` defined as the Hamming weight of a byte, shell populations are `64 * C(6, s)` for `s = 0..6`:

```
pops = [64, 384, 960, 1280, 960, 384, 64]
```

These sum to 4096 with mean shell `⟨S⟩ = 3.000`. Holographic balance `|H|^2 = |Ω|` holds exactly. Rank-by-rank reachability under fiber-complete restriction is

```
r   |Reach|   θ = |Reach|/|Ω|   note
0       2     0.000488          not fiber-complete (gauge doublet)
1       4     0.000977          |Reach| = (2^1)^2
2      16     0.003906          |Reach| = (2^2)^2
3      64     0.015625          |Reach| = (2^3)^2
4     256     0.062500          |Reach| = (2^4)^2
5    1024     0.250000          even-weight plateau (parity-obstructed)
6    4096     1.000000          full manifold
```

The rank-5 plateau follows directly from the parity functional `parity(q) = popcount(q) mod 2`, which is a homomorphism from the transport group to GF(2) whose kernel is the rank-5 subspace of even-weight transport values. A rank-5 set confined to that kernel drives only even-weight transport, so from the shell-6 anchor it reaches only even shells and the cluster closes at `32^2 = 1024`. Full reachability therefore requires odd-shell access beyond the mere rank condition `r = 6`, and the same parity cohomology class is what separates `E_span` from `E_full` in the coverage hierarchy.

### 3.2 The Exact Coverage Fraction θ(p)

The percolation parameter `p` is the independent probability that each of the 256 byte operators is included in the allowed set `A`. Restricting the byte alphabet degrades the transport rank on the chirality register, and the reachable cluster shrinks as the square of the surviving root. For the micro-reference payload protocol, the full coverage distribution admits a closed form

```
θ(p) = ∑_k P(rank = k) · (2^k)² / 2^(2d)
```

The sum runs over the exact rank probability mass function `P(rank = k)`, with chirality dimension `d = d_χ` (physical instance `d_χ = 6`) and coverage fraction `θ = |Reach|/|Ω|`. Physical coverage uses the conditional form for a nonempty generator set. Exact coverage at the hierarchy thresholds (unconditional audit form) is

```
p       θ(p)      event
0.0219  0.025530  E_span (weak transport, p/Δ ≈ 1.04)
0.0273  0.043530  E_full (strong, r = 6)
0.0402  0.112578  E_spectrum (defect completion)
0.0908  0.579325  P(rank = d) = 1/2 (micro-ref p_c)
0.3086  0.999831  E_word (holonomy transport)
```

Coverage saturates by `p ≈ 0.30`, which is a property of the kernel graph and is independent of the electrostatic barrier. At the exact micro-reference rank threshold, `θ(p_c_rank = 0.0908) = 0.5793`, a fuel-independent value defined at `p = p_c`.

The rank-ladder thresholds are exact. For integer rank `r`, `p_c(r)` is the inclusion probability at which the reachable set first reaches rank `r` (the probability that the rank equals `r` is one half). Distinct values after de-duplication are

```
p_c(rank ladder) = {0.293, 0.219, 0.146, 0.091}
```

with the exact micro-reference rank threshold `p_c(rank) = 0.0908`. The ratio `p_c(span) / Δ ≈ 1.056` matches the compact-geometry target near 1.04.

### 3.3 The Coverage Hierarchy

The coverage events are successively stronger conditions on the same rank-six root. Full transport rank `r(A) = 6` is necessary for full reachability but not sufficient for the finer structure the fusion module reads, because a root can be full-dimensional while remaining sparsely populated, anisotropically branched, or uncomposed into closure operators. Each additional event demands one more of these properties, so the events turn on at separable generator fractions as `p` increases.

```
p_c(span)     = 0.0219   at least one path from horizon 6 to horizon 0
p_c(full)     = 0.0273   full transport rank r = 6 with odd-shell access
p_c(spectrum) = 0.0402   all seven transport-defect weights present
p_c(rank)     = 0.0908   exact micro-ref rank threshold, P(rank = d) = 1/2
p_c(word)     = 0.3086   four-byte closure words available (holonomy transport)
```

Span requires only that the reachable set touch the opposite horizon, which can occur along a low-dimensional transport subspace, whereas full reachability adds the requirement of odd-shell transport. Spectrum completion requires the root to be uniformly covered by all seven defect weights, exceeding the spanning condition. The word event is the strongest condition, requiring the root to be composed into four-byte closure operators that carry holonomy, and its availability follows `1 - (1 - p^4)^64` because each closure word needs four independent byte inclusions. The ordering `p_c(span) < p_c(full) < p_c(spectrum) < p_c(rank) < p_c(word)` therefore reflects increasing structural demand on a single fixed root.

### 3.4 Protocol Sensitivity

Under the default fusion model the inclusion probability that feeds `θ` is the Δ-dial value `p_Δ = E / V_b`. A second protocol, generator inclusion by q6 payload, produces a parallel coverage curve. At sample D-T energies the two protocols give

```
E_keV    τ       T        p_Δ     θ_micro     θ_q6
10.0     9.213   0.0001   0.0225  3.569e-02   1.239e-02
20.0     6.038   0.0024   0.0450  1.558e-01   7.795e-02
30.0     4.632   0.0097   0.0676  3.582e-01   2.307e-01
50.0     3.221   0.0399   0.1126  7.481e-01   6.359e-01
72.5     2.399   0.0908   0.1633  9.436e-01   9.049e-01
100.0    1.801   0.1651   0.2252  9.941e-01   9.888e-01
```

At `E_rank = 72.5` keV, where bare transmission reaches `p_c(rank)`, both protocols already sit above `theta = 0.90`. The fusion calculations use the micro-reference protocol with conditional coverage.

### 3.5 Discrete Grammar and Continuous Observables

The discrete kernel fixes the grammar through the classes `(k, ℓ)`, the thresholds `p_c(r)`, and the coverage `θ(p)`. Continuous observables enter as ruler readings. Once `Δ` and `v` are fixed, `n(E)` is a smooth function of `E`. Once the exact rank distribution is fixed, `θ(p)` is a smooth function of `p`. The kernel supplies the combinatorial structure and the energy coordinate supplies the continuity. A given nucleus or resonance is predicted to land at a specific tick on a ruler whose unit is fixed by the W/Z ratio. The falsifiable quantity is the tick residual.

## 4. Forced Minimum Nuclear Excitation

### 4.1 Minimum Isomeric Excitation

The primary prediction of this section is the minimum half-life-tagged, optically addressable nuclear excitation on the forced grammar class `(k, ℓ) = (6, 2)`:

```
E_min = v · ρ² · Δ⁶ / √5 · 2^(C3 Δ²)
```

This is the energy of the forced nuclear class: electroweak anchor `v`, spinorial dress `ρ²`, sixth aperture grade `Δ⁶`, STF equipartition `1/√5`, and equatorial tick `2^(C3 Δ²)` with enumerator weight `C3 = C(6, 3) = 20`. What the ENSDF half-life-tagged eV-band census can falsify is the absence of any such isomer below the tolerance window around `E_min`. The stronger reading, that this residual is the absolute minimum nuclear excitation of the ground-state sector, is a physics interpretation of the Δ⁶ W-boundary that extends beyond the census check.

The derivation uses only upstream quantities. The electroweak sector closes at `Δ⁵` in the compact-geometry five-order expansion, which is the BU-balanced ground relative to the representation boundary. The sixth grade is the W-channel representation boundary, the unique full-flag K4 endpoint. The first excitation beyond ground is that residual. Any lower structure would have been required to close at `Δ⁵` (`docs/CGM_Logic.md`). The inputs are fixed. `Δ` is the value recovered in Section 2.2, and `(k, ℓ) = (6, 2)` is the forced nuclear class.

Numerically,

```
E_min = 8.3563 eV
```

### 4.2 Verification Against Th-229m

The lowest established optically addressable nuclear excited state is the Th-229m isomer at 8.3557335(8) eV in CaF2 (Zhang et al.). This is more than 10^5 times lower than typical nuclear excitations in the keV-MeV range, and it is the unique known nuclear excitation in the laser and VUV window.

The forced prediction and the measurement agree to a relative error of 7.19e-05, with a ruler residual of 0.005 ticks against the forced class `(6, 2)` (tick tolerance 0.1). The Level assigns Th-229m to `(6, 2)`. The prediction contains zero free parameters and has grammar rank 1. The forced-class energy window at tolerance `tol` ticks is

```
E_lo = E_min / 2^(tol · Δ) ,   E_hi = E_min · 2^(tol · Δ)
```

so `tol` is the allowed tick half-width about `E_min` and `[E_lo, E_hi]` is the corresponding energy band. At `tol = 0.1` that band is `[8.3444, 8.3683]` eV. The ENSDF eV-band isomer census in `data/catalogs/ensdf/ensdf_ev_band_levels.csv`, filtered to half-life-tagged entries via the IAEA LiveChart levels API (ENSDF underlying evaluations in `data/catalogs/ensdf/SOURCE.txt`, 214 actinide level files), contains no isomer below that window. The prediction is therefore the empirical minimum among known excitations. Census status on the filtered band is

```
status         label                   E_eV     near(k,ℓ)  tick
PASS           Th-229m (Zhang CaF2)    8.3557   (6,2)      +0.005
UNCLASSIFIED   U-235 ENSDF/iso        76.0000   (6,0)     -95.283
```

The census yields one passing entry and one unclassified entry. The null probability that a random energy in the eV band `[0.1, 200]` eV lands within `tol` ticks of `E_min` under log-uniform measure is

```
p_one = (2 · tol · Δ) / log2(band_max / band_lo)
```

Writing `band_lo = 0.1` eV and `band_max = 200` eV for the census window, and with census size `N = 1` in the forced window, the probability of at least one hit is `1 − (1 − p_one)^N = 0.0004`. The ENSDF Adopted listing for Th-229 still records approximately 7.6 eV. That value is superseded here by Zhang for tick checks. The Pu-239 first excitation lies above 1 keV, consistent with the prediction.

### 4.3 The Strong Bare Scale

The strong bare scale is the other forced anchor that feeds the holographic product and the fusion module:

```
E_str = v · Δ³
```

Numerically `E_str = 2.1838 MeV` once the locked `v` and `Δ` are substituted. The minimum excitation is related to it by the holographic product

```
E_min = E_str · (ρ² · Δ³ / √5) · 2^(C3 Δ²)
```

so the nuclear residual is the strong bare scale dressed by two holonomy layers, one further aperture grade, the STF factor, and the equatorial tick. The strong bare scale is the anchor on which the Coulomb barrier and the tau-dial (formalized in Section 7.2) are built.

### 4.4 Spectral Bridge from the Wavefunction Kernel

The nuclear residual inherits the spectral structure that recovers the electroweak masses. The wavefunction is the state on the finite kernel manifold,

```
H = l²(Ω),   dim Ω = 4096,   |horizon| = 64
```

with shell-number operator `D_shell` whose reduced spectral moment is `M_shell = Tr(D_code) = 192`. The four electroweak channels are the K4 operator group `{id, W2, W2', F}`, where K4 denotes the Klein four-group reached by byte words on `Ω`. Cumulative fold-crossing depth fixes each channel's flag tuple `(base, rot, bal)`:

```
channel   operator   flags (base, rot, bal)
Top       id         (0, 0, 0)
Higgs     W2         (1, 0, 0)
Z         W2'        (1, 1, 0)
W         F          (1, 1, 1)
```

The W channel is the full-flag endpoint and carries the largest positive sixth-grade residual. Each mass is a spectral expansion in the aperture,

```
L_i(Δ) = a_i·Δ + b_i + c_i·Δ² + p_i·(Δ/√5)·Δ² + q_i·Δ⁴ + r5_i·Δ⁵
m_i = v / 2^{L_i(Δ)}
```

Channel index `i` labels Top, Higgs, Z, or W. The polynomial `L_i` gives the tick coordinate and `m_i` is the absolute mass. The coefficients are fixed by kernel algebra. Recovered masses are

```
Top    172.7600 GeV
Higgs  125.1000 GeV
Z       91.1876 GeV
W       80.3790 GeV
```

The W/Z ratio from the spectral expansion is `0.88146853`, matching the PDG value, with absolute error on the recovered `Δ` equal to `2.203e-11`. Carrier traces `C(q)` enter the channel corrections and the nuclear matrix-element proxies.

`E_min` is the nuclear residual of this structure. The sixth grade is the W-channel representation boundary, so the nuclear scale inherits the kernel spectral anchor rather than introducing a new parameter. The 32-bit spinorial lift closes through K4 algebra, Gate F (a shell-preserving involution on the carrier, `F² = id`), and exact rank-lock (PMF match `0.999667`).

## 5. Deuteron Binding: Strong Bare Plus Tensor

### 5.1 The Two-Term Decomposition

The deuteron binding energy is reconstructed from the strong bare scale and a tensor correction:

```
E_d = v · Δ³ + v · Δ⁴ · (2 / √5)
```

The first term is the bare strong scale `E_str ≈ 2.1838 MeV`. The second is the tensor correction, whose coefficient `2/√5` is identical to the W/Z p-charge difference `(p_W − p_Z)/√5` in the electroweak expansion. That coefficient is the discrete trace-free quadrupole correction from the kernel grammar that fixes the electroweak masses.

Using the kernel constants,

```
E_bare   = 2.1838 MeV
E_tensor = 0.0404 MeV
E_total  = 2.2242 MeV
```

The measured deuteron binding is 2.2240 MeV (Particle Data Group few-nucleon summary, Navas et al., Phys. Rev. D 110, 030001, 2024), so the full formula closes to a relative error of 8.89e-05, well within the 5e-04 threshold. The bare term alone has relative error 1.81e-02, which the tensor correction removes.

### 5.2 The Tensor Fraction and the Discrete Pion

The tensor term is 1.82 percent of the total binding. In the discrete CGM frame the tensor correction `v · Δ⁴ · (2/√5)` arises from the Δ-shell-2 isospin-flip carrier move, the minimal spin-0, isospin-1 carrier excitation. That move is the discrete counterpart of the pion, the Goldstone boson of chiral symmetry breaking in the continuous theory. The Δ⁴ gap, the step from the bare `Δ³` to the tensor `Δ⁴`, sets the chiral-symmetry-breaking scale in the discrete frame. The carrier quantity `C(2) = 7/3` is the trace of that move. This identification is a structural parallel between the discrete carrier move and the continuous Goldstone mode.

### 5.3 The Level Residual of the Deuteron

The Level assigns the deuteron to the strong bare class `(3, 0)` at a ruler residual of minus 1.272 ticks. This residual is the code-atom resolution limit of the underlying kernel lattice, the discrete set of named code atoms `{C1, C2, C3, halves, differences, sums}` on the ruler, among which no 1.272-tick code atom exists. The binding formula closes to 1e-04, so the residual is a discretization artifact of the Level readout. The deuteron sits off the integer grid while Th-229m sits on it, and both placements are consistent with a discrete lattice of named code atoms.

## 6. Alpha and Beta Decay on the Kernel Carrier

### 6.1 The Three Element-Changing Paths

Nuclear structure contains three element-changing transitions. Alpha decay changes charge by minus two. Beta decay changes charge by plus one. Fusion combines two nuclei. This section treats alpha and beta. Fusion follows in Section 7.

The kernel encoding used here has three layers, introduced in the order an external reader meets them.

First, the nuclear quantum numbers. Charge `Z`, neutron count `N`, spin `J`, and parity determine a chirality shell and a spin-parity companion. The chirality shell is `|N − Z| mod 7`. It is stored in a six-bit register `χ6`. Orientation is built from the nuclear data before the Hamming weight is forced to that shell:

```
χ_rot = (Z mod 8) ⊕ (2J mod 8)     (Frame-0 bits 0–2)
χ_tr  = (N mod 8) ⊕ parity_bit       (Frame-1 bits 3–5)
χ6    = set_weight(χ_rot | χ_tr, |N − Z| mod 7)
```

Here `parity_bit` encodes the parity. The oriented six-bit word is then forced to weight `|N − Z| mod 7`, so the shell of the encoded nucleus matches the nuclear shell formula.

Second, the spin-parity companion. An eight-bit intron carries `2J` in six payload bits plus one family-high bit (capacity `2J ≤ 127`) and parity in the family-low bit. Every operator byte is compared to a fixed eight-bit reference pattern, the micro archetype `GENE_Mic = 0xAA`, by the transcription

```
intron = byte ⊕ GENE_Mic
```

so the intron is the mutation of the byte relative to that archetype. The nuclear intron used above is that same companion word, already filled from `J` and parity.

Third, the macroscopic carrier. Charge and chirality assemble a 24-bit state, the GENE_Mac tensor, from the pieces `u6 = Z mod 64` (parity bit optionally set) and `v6 = u6 ⊕ χ6`. That 24-bit tensor is the carrier on which decay operators act. Alpha and beta transitions are byte words on the kernel graph that update GENE_Mac. The intron carries the spin-parity content of the full atom.

### 6.2 Alpha Decay: Gate F, Shell-Preserving

Alpha emission ejects a `^4He` cluster (`N = Z`), so `N − Z` is conserved and the daughter stays on the same chirality shell as the parent. The operator is the four-byte Gate F word, with `Gate F = W2 ∘ W2′`. Gate F is the global-inversion element of the Klein four-group `{id, S, C, F}` on the carrier (`hQVM_Features_Report.md`). It is an involution that preserves chirality and therefore preserves shell (verified on 200 of 200 sampled states) while flipping the Z2 carrier sheet. Applying F twice returns the carrier to rest. The explicit four-byte construction for each K4 family index is recorded in `hQVM_Specs_Formalism.md`.

The bulk census over the IAEA LiveChart ground states (`data/catalogs/ensdf/iaea_livechart_ground_states.csv`, 2572 catalog entries with usable `J, P`) reports 314/314 for all three metrics on the 314 alpha parents with a catalogued daughter, confirming that Gate F preserves shell, shell-parity, and the daughter `|N−Z| mod 7` formula:

```
shell preserved (Gate F) .... 314/314
shell-parity conserved ...... 314/314
shell = |N-Z| mod 7 daughter  314/314
```

The alpha half-life is assembled from the tau-dial tunnel transmission `T = exp(−τ)` of the alpha on the daughter barrier, the carrier-trace hindrance `H_L = C(L) / C(0)`, the assault frequency `ν = 10^21 s^−1`, and the structural preformation `P_α = 5 / 2^20 = 4.7684e-06`,

```
T½ = ln 2 · P_α / (ν · T · H_L)
```

`P_α` counts the five bulk STF shells against the operator-state phase space `|Ω| · |Alphabet| = 2^20`. The factor `T` is the tunnel transmission and `H_L` is the angular-momentum hindrance at multipolarity `L`.

For Th-229 → Ra-225 the Gate F word is `(0x96, 0x97, 0x16, 0x17)`. The carrier maps `0xaaa555 → 0x555aaa` with shell 6 preserved. With `Q_α = 5.168 MeV`, `L = 2`, tunnel factor `T_tunnel = 4.135e-38`, and `H_L = C(2)/C(0) = 1/3`, the structural half-life is `2.397e11 s` against the measured `2.498e11 s` (ratio 0.96, residual −4.0 percent).

Across 310 alpha parents that carry both Q-value and half-life in the catalog, the structural estimator yields

```
ratio within [0.5, 2] ..... 113/310
ratio within [0.1, 10] .... 271/310
median ratio .............. 0.697
```

Sample closures include Nd-144 → Ce-140 (ratio 1.81) and Sm-146 → Nd-142 (ratio 1.10). The estimator is a structural lower bound from tunnel transmission and carrier hindrance. Absolute rates still require the ordinary nuclear preformation physics that sits outside the kernel rationals.

Spot checks on five alpha parents (Th, U, Ra, Po, Pu chains) all preserve shell and shell-parity under Gate F.

### 6.3 Beta Decay: Single-Byte Shell Advance

Beta-minus decay (`n → p`) changes `N − Z` by minus two at fixed mass number `A`, so the chirality shell advances by two. The transition is a single operator byte on the carrier, in contrast to the four-byte Gate F word of alpha decay. Three beta branches flip the intron bit pair that controls spin change, namely bits 1 and 6 of `intron = byte ⊕ GENE_Mic`:

```
0x29   bit 1 only      ΔJ = +1
0x6b   bit 6 only      ΔJ = −1
0x69   bits 1 and 6    ΔJ = 0   (isospin advance |Δshell| = 2)
```

Bit 1 contributes `+1` to `ΔJ` and bit 6 contributes `−1`, so setting both yields `0`. The byte advances the carrier by two chirality shells with no Δ-ruler mass step.

The bulk census over 801 beta-minus parents from the same LiveChart ground-state catalog reports:

```
parent J round-trip .......... 801/801
shell-parity conserved ...... 801/801
decoded J-rule vs parent .... 801/801
catalog |dJ| <= 1 ........... 402/402
daughter-shell closure ...... 801/801
```

All 801 cases pass the round-trip, parity, and closure checks. The gated claims are distinct:

```
claim                                      domain                         result
shell-parity conserved                     β− parents with catalog daughter  801/801
daughter shell reachable (FWD/REFL/SRCH)   same                             801/801
parent J round-trip                        same                             801/801
daughter J for |ΔJ|≤1 (allowed stratum)    depth-1 subset                   402/402
daughter J vs catalog (all depths)         all 801                          402/801
```

The 402/801 overall J figure is the full catalog including higher-|ΔJ| compositions outside the allowed stratum. On the depth-1 stratum `|ΔJ| ≤ 1` the agreement is 402/402. The daughter spin is operator-emitted (`J → J + dJ`) in the intron. The parent GENE_Mac tensor is a two-to-one projection of the full atom, so branch provenance lives in the intron or word composition.

For tritium (`³H → ³He`) the intron is `0xc3`, the byte is `intron ⊕ GENE_Mic = 0x69`, and the carrier maps `0xaaa555 → 0xaaa956` (shell 6 → 4, `|Δshell| = 2`). The half-life estimate uses the ordinary Fermi integral and an empirical superallowed anchor,

```
f(Z, Q) = ∫₁^{W₀} F(Z,W) · p · W · (W₀ − W)² dW ,   T½ = ln 2 · ft / (f · |M|²)
```

Endpoint energy enters as `W₀ = (Q + m_e)/m_e`. The Fermi function is the nonrelativistic form `F = 2πη/(1 − e^{−2πη})` with Sommerfeld parameter `η = α Z W / p`. Kernel matrix element and comparative half-life are fixed at `|M|² = 1.0` and `ft = 10^{3.05}`. With `Q_β = 18.591` keV (LNHB) and `f(Z=2, Q) = 2.880e-06`, the estimated half-life is `2.700e8 s` against the measured `3.885e8 s` (ratio 0.69).

Decoded daughter-J agreement with the catalog is `402/801` overall. On the depth-1 subset `|dJ| ≤ 1` the agreement is `402/402`. Branch-shell match on any of the three beta branches is `87/801`, and joint shell-plus-J match is `44/801`. Those lower rates follow because GENE_Mac alone leaves the branch underdetermined until the intron is fixed.

### 6.4 The Daughter-Shell Closure and Branching Depth

The daughter shell is routed deterministically by three classes of move, which together close all 801 beta-minus parents:

```
FWD  (3 beta branches) ......... 243/801
REFL (W2 byte 0x2A, w -> 6-w) .. 70/801
SRCH (derived byte, XOR-transport rule)  488/801
CLOSED total ..................... 801/801
```

The SRCH byte is derived by a kernel-grammar rule. It selects the smallest-|q| beta-family byte whose transport mask `q` satisfies `|χ ⊕ q| = wp + |q| - 2|χ & q| = daughter_shell`. The closure therefore follows from that rule. The shell-transport identity verified on all 64 χ words is `|χ'| = |χ| + |q| - 2|χ & q|`, and the overlap `|χ & q|` makes the shell change state-dependent. The Hamming ladder carries no mod-7 cycle. Wrap cases use W2 reflection `w ↦ 6 − w`.

The REFL residue (70 parents) consists of cases whose daughter shell equals `6 − wp`. Representative parents include Be-12, B-14, C-16, N-18, Na-26, Mg-28, and Si-32. The SRCH residue (488 parents) requires a derived beta-family byte outside the three beta branches. Representative parents include n-1 (`need_byte = 0x04`), H-3 (`0x03`), He-6 (`0x10`), Li-9 (`0x04`), and C-14 (`0x27`). Derived-byte `dJ` matches the catalog `dJ` on `145/801` cases overall and on `145/402` of the depth-1 subset.

Branching by catalog `|ΔJ|` reads the root-coverage depth of the beta sector. A single beta half-cycle emits `ΔJ` in `{-1, 0, +1}`, the depth-1 stratum. Larger catalog `|ΔJ|` are compositions of half-cycles, so higher `|ΔJ|` sits at deeper strata:

```
depth-1 (|dJ| <= 1) ......... 402/801
depth-2 (|dJ| <= 2) ......... 198/801
depth-3+ (|dJ| > 2) ........ 201/801
catalog dJ resolved at depth <= 2 ... 600/801
```

The catalog `|ΔJ|` histogram splits as `0: 139, 1: 263, 2: 198, 3: 98, 4: 48, 5: 24, 6: 12, 7: 9, 8: 8` (plus two outliers at 23). The two largest low-`ΔJ` bins (139 + 263 = 402) match the 2:1 prediction of the two-reference beta-branch family at ratio 1.89:1. Beta branching and percolation depth are the root coverage that the Square reads in the fusion module.

Spot checks on four beta parents (H-3, C-14 family, Co-60 family, Sr/Y chain) all close the daughter shell under the FWD/REFL/SRCH routing.

### 6.5 Carrier Traces as Beta Matrix Elements

The kernel shell-transition matrix `M_q` (Krawtchouk shell-mixing on the six-bit chirality register) supplies exact rational carrier traces. For even `q` the diagonal trace is nonzero and equals `C(q) = Tr(M_q) = 7/(q+1)`. For odd `q` the diagonal vanishes and `C(q) = Tr(M_q²)` is the return-trace:

```
C(0) = 7
C(1) = 28/9
C(2) = 7/3
C(3) = 52/25
C(4) = 7/5
C(5) = 28/9
C(6) = 1
```

The Fermi proxy is `|M_F|^2 = C(0) = 7`. The Gamow-Teller proxy is `|M_GT|^2 = C(1) = 28/9`. The forbidden ladder is `C(3)/C(1) = 117/175` and `C(5)/C(1) = 1`. The alpha hindrance is `H_L = C(2)/C(0) = 1/3`, the discrete result for an `L = 2` transition. These are kernel rationals. The Fermi integral `f(Z, Q)` and the superallowed `ft` that convert them to a half-life are ordinary nuclear physics, as demonstrated in the tritium estimate above.

### 6.6 Nuclear Magic Numbers from the Carrier Algebra

Nuclear magic numbers are nucleon counts at which a closed shell forms in an independent-particle description of the nucleus. In this subsection a closed shell is identified by a large energy gap between the highest occupied single-particle level and the lowest unoccupied level in a spectrum built from the carrier algebra. The single-particle Hamiltonian uses coupling constants already fixed by kernel invariants. The seven canonical spherical magic numbers 2, 8, 20, 28, 50, 82, and 126 emerge as large-gap closures in the mixed Nilsson spectrum at the derived point `(κ, μ) = (1/32, 1/5)`.

The Mayer-Jensen shell model accounts for the harmonic oscillator closures 2, 8, and 20 through independent-particle filling of a central potential. The additional closures 28, 50, 82, and 126, often called intruder magic numbers, require a spin-orbit term strong enough to place the aligned branch `j = l + 1/2` below `j = l − 1/2`. In the Nilsson deformed oscillator, quadrupole mixing enters through the deformation weight `μ`, and both `κ` and `μ` are ordinarily treated as continuous parameters fit to reproduce the observed closure set. The present construction keeps the standard Nilsson Hamiltonian form but derives the sign of the spin-orbit term, the values of `κ` and `μ`, and the Δn = 2 mixing rule from the SE(3) carrier algebra. The spin-orbit inversion is therefore read as a kinematic consequence of left chirality on the discrete spatial shadow, not as an independently postulated dynamical coupling tuned after the fact.

The six payload bits of GENE_Mac are the six generators of the Euclidean group SE(3) in three dimensions. Three bits belong to Frame 0 and carry the rotational content of the algebra. Three bits belong to Frame 1 and carry the translational content. Frame 1 therefore supplies the three spatial modes of a three-dimensional harmonic oscillator. A shell of total quantum number `n` contains `(n+1)(n+2)/2` spatial orbitals. Frame 0 supplies spin one-half, so each spatial orbital `(n, l)` splits into the two branches `j = l + 1/2` and `j = l − 1/2` with degeneracy `2j+1`, subject to the oscillator selection rule that `l` and `n` share parity.

The CGM framework assigns a preferred orientation to this ordering. The Common Source stage introduces a left-handed bias that fixes chirality and ancestry throughout the construction. Within each oscillator shell, the sign of that bias determines which `j` branch is filled first when levels are ordered by increasing energy. Under the left-chiral assignment required by the CS axiom, the aligned branch `j = l + 1/2` sits lower than `j = l − 1/2`. This ordering is consistent with the same left-biased ancestry preservation that governs the decay routing of Sections 6.2 through 6.5. When chirality is reversed, the intruder set 28, 50, 82, and 126 no longer appears among the large-gap closures, and only the harmonic oscillator remnant 2, 8, and 20 remains dominant in the gap ranking.

Before the energy spectrum is evaluated, the left-chiral fill order provides a structural guide to where intruder closures must land. If the highest-`j` subshell within each major shell is filled first, the cumulative counts 28, 50, 82, and 126 appear together with the oscillator closures 2, 8, and 20. Under right chirality the highest-`j` subshell is filled last and those intruder counts move to the full-shell boundaries 40, 70, 112, and 168. This ordering exercise anticipates the intruder set but does not replace the gap definition. The authoritative identification of a magic closure is a large adjacent gap in the filled single-particle spectrum. The canonical set therefore splits into a chirality-invariant harmonic-oscillator subset {2, 8, 20} and a chirality-selective intruder subset {28, 50, 82, 126}.

The kernel exposes the radial and angular content on which the spectrum acts. The six-bit chirality register has popcount equal to the radial shell index `s` running from 0 to 6. The population of shell `s` on the carrier is `64·C(6, s)`, where `C(6, s)` is the binomial census of the Hamming scheme H(6, 2). That census satisfies `|Shell_s| = |Shell_{6−s}|`. Within each shell the in-shell multiplicity is 64, the discrete counterpart of angular multiplicity at fixed radial shell. Gate F, introduced in Section 6.2, preserves the radial shell index while exchanging the two carrier sheets, and therefore acts within the angular coordinate without changing the radial census.

The Nilsson Hamiltonian is written in oscillator units with `ℏω` set to unity,

```
E(n, l, j) = (n + 3/2) − κ ⟨L·S⟩_j − κ μ ⟨l²⟩
```

The oscillator quantum number `n` is the major shell index. The orbital angular momentum quantum number `l` satisfies the oscillator parity constraint that `l` and `n` share parity and runs over the allowed values in shell `n`. The total angular momentum quantum number `j` takes the values `l + 1/2` and, when `l ≥ 1/2`, `l − 1/2`. The spin-orbit expectation value is

```
⟨L·S⟩_j = ½[j(j+1) − l(l+1) − 3/4]
```

where `j` and `l` are the total and orbital angular momentum quantum numbers of the subshell. The quadrupole deformation invariant is

```
⟨l²⟩ = l(l+1)
```

where `l` is again the orbital angular momentum quantum number.

The parameter `κ` sets the overall scale of the spin-orbit and deformation terms in oscillator units. The parameter `μ` sets the relative weight of the deformation term. The CS axiom fixes the sign of the spin-orbit term so that the aligned branch `j = l + 1/2` lies lower under left chirality. The deformation weight is identified with the reciprocal of the symmetric trace-free bulk dimension, which equals five, giving `μ = 1/5`. That dimension is the five independent spatial modes of the `l = 2` quadrupole on the SE(3) shadow. The spin-orbit scale is identified with the BU dual-pole monodromy expressed as a fraction of a full turn. Writing `τ = δ_BU/(2π)`, the coupling is taken as the 256-tick turn quantization of `τ`, where `Q_256` denotes rounding to the nearest 1/256-turn tick, which gives `κ = Q_256(τ) = 1/32`. Both assignments use only quantities already fixed by the CGM constants chain of Section 2. The same STF bulk dimension enters the electroweak mass expansion through the `1/√5` equipartition factor of Section 2.1, and the same monodromy ratio `δ_BU/m_a` that defines the aperture gap Δ also supplies τ. The couplings that govern shell closure at `(Z, N) = (82, 126)` therefore belong to the same geometric ratio system as the W/Z mass split, not to a separate nuclear parameter set.

Two spectra are evaluated at this anchor. In the diagonal spectrum each orbital receives the energy above without cross-shell mixing. In the mixed spectrum the quadrupole operator introduces Δn = 2 couplings between orbitals of the same `l`, `j`, and `m` whose major quantum numbers differ by two, where `m` is the magnetic substate label, the projection of `j` onto a fixed axis. The mixed spectrum is solved on the `m`-substate basis so that level counting is explicit after diagonalization. A closure is recorded when an adjacent gap exceeds 1.8 times the local median spacing, and gap prominence ranks the largest closures relative to their neighbors.

The diagonal spectrum yields strong gaps at the harmonic oscillator closures 2, 8, 20, 40, 70, and 112. It does not promote the intruder closures 28, 50, 82, and 126 to the dominant gaps. The mixed spectrum changes this ranking. At `(κ, μ) = (1/32, 1/5)` every canonical magic number appears in the closure set under the gap criterion, and the four intruder numbers move from subdominant structure in the diagonal ordering to dominant gaps in the mixed prominence ranking. Intruder prominence is therefore mixing-generated at this anchor: the diagonal central-plus-spin-orbit model alone does not place them among the top closures, while Δn = 2 quadrupole mixing does.

The absolute spacing of the oscillator steps is carried by the strong bare scale

```
E_str = v·Δ³ ≈ 2.18 MeV
```

introduced in Section 4.3, where `v` is the electroweak vacuum expectation value and Δ is the aperture gap of Section 2. The BU aperture fraction Δ supplies a natural fractional splitting scale on that anchor, which corresponds to an energy scale of order `E_str·Δ ≈ 45 keV` when expressed as a single factor on the strong bare scale.

Empirical checks are consistent with this reading. Every doubly magic nucleus in the standard list has ground-state spin-parity 0+ and a chirality-shell index `|N − Z| mod 7` that lies in the central or paired shells of the `C(6, ·)` census. For lead isotopes near `Z = 82`, the IAEA LiveChart catalog reports a binding energy per nucleon. Total binding is reconstructed as

```
B_tot(Z, N) = binding(Z, N) · (Z + N)
```

where `Z` is the proton number, `N` is the neutron number, and `binding(Z, N)` is the catalog entry in keV per nucleon. The two-neutron separation energy is

```
S_{2n}(Z, N) = B_tot(Z, N) − B_tot(Z, N − 2)
```

and the curvature indicator is

```
δ_{2n}(Z, N) = S_{2n}(Z, N) − S_{2n}(Z, N + 2)
```

with `S_{2n}` and `δ_{2n}` reported in MeV. The `S_{2n}` curve falls and `δ_{2n}` peaks at `N = 126`, with `δ_{2n} ≈ 5.0 MeV` at that closure, which is the expected shell signature for a neutron magic number. Extending the mixed Nilsson basis to twelve major shells places closure candidates in the superheavy region that include 114, 120, 126, and 184.

Future analyses may attach residual nucleon-nucleon interactions, pairing correlations, and collective degrees of freedom to this single-particle skeleton. A full conversion of the atomic spectroscopic catalog into the same closure language remains a separate trestleboard task aligned with Section 6.7. At the CGM anchor the closure pattern separates three contributions. The harmonic-oscillator closures are kinematic. The intruder closures are chirality-selective in fill order and in the spectral sign of the spin-orbit term. Their dominance in the gap ranking is mixing-generated. The couplings `κ` and `μ` are fixed by kernel invariants rather than fit to the magic numbers. Together these readings place the seven canonical magic numbers as large-gap closures of the same carrier algebra that routes alpha and beta decay.

### 6.7 Atomic Spectroscopy Parallel

Same-element spectral line pairs align to compact-geometry code levels on the ruler. Conversion of the full spectroscopic catalog into a self-check lies outside the scope of this analysis. The measured alignments reported with the compact-geometry findings include

```
level   compact role                        best pair         err(ticks)
12      constitutional diameter             He 10917/12968    0.001
16      mask-code weight 2                  Cs 8047/10124     0.001
32      mask-code weight 4                  Na 2839/4494      0.000
48      mask-code weight 6 / four-byte word Na 3094/6161      0.008
64      mask-code weight 8 / |H|            Cs 5466/13693     0.006
80      mask-code weight 10                 Na 2905/9154      0.001
96      mask-code weight 12                 He 4713/18685     0.001
```

Antihydrogen mirror-tick sensitivity is recorded as `η_X = log2(ν_H / ν_Hbar) / Δ` with sigma-tick scale `9.4e-3`. These alignments sit on the atomic/deep-IR sector of the Square and are structural parallels to the nuclear placements.

## 7. The Fusion Module

### 7.1 The Coulomb Barrier as a Placed Grammar Coordinate

Fusion of light nuclei is exothermic on the rising flank of the nuclear binding-energy curve, with ^4He among the most tightly bound products. Before the short-range nuclear attraction can act, the nuclei must approach through the long-range Coulomb repulsion. The classical barrier height for that approach is the Coulomb barrier

```
V_b = 1.44 · Z1 · Z2 / r_fm ,   r_fm = 1.2 · (A1^(1/3) + A2^(1/3))
```

Charge numbers are `Z1`, `Z2` and mass numbers `A1`, `A2`. Energies are in MeV and radii in fm. Quantum tunneling allows fusion at kinetic energies below `V_b`. Gamow (1928) first applied tunneling to alpha decay and then to fusion as the inverse process, and Atkinson and Houtermans (1929) used that penetration to estimate stellar fusion rates. The Gamow energy of the reduced-mass two-body problem is

```
μ = A1 · A2 / (A1 + A2) · m_N ,   E_G = 2 μ (π α Z1 Z2)²
```

Reduced mass `μ` uses nucleon mass `m_N = 931.494 MeV` and fine-structure constant `α = 1/137.036`. The Gamow penetration factor is then `P_Gamow(E) = exp(−√(E_G / E))`. The trestleboard takes `V_b` from the formula above (default `r0 = 1.2` fm) and reads it as a placed coordinate on the strong-family ladder. Its ruler tick `n(V_b)` lands on a class with `k = 3` (the strong bare scale `v · Δ³`), and the dress rank `ℓ` varies with `Z1 Z2` so that heavier charge products sit lower on the strong ladder.

The barrier-placement test evaluates two claims:

1. The barrier tick `n(V_b)` lands on a strong-family class (`k = 3`).
2. On a fine energy grid below the barrier, the truncated-barrier transmission

```
τ_b(E) = 2π · √(E_G / E) · (1 − √(E / V_b)) ,   s(E) = (1/E) · θ(E/V_b) · exp(−τ_b)
```

attains its maximum at an energy whose tick coincides with `n(V_b)` within tolerance (7 ticks, about 10 percent in energy). The optical depth is `τ_b`. The score `s(E)` weights coverage `θ` by the usual Gamow-like exponential. Because `τ_b → 0` as `E → V_b`, the peak sits near the barrier by construction of the truncated form. The nontrivial grammar claim is the `k = 3` placement of `V_b` itself.

Per-fuel barrier placement is

```
fuel      Z1Z2   V_b(MeV)  n(V_b)   n_peak  Δn     class (k,ℓ)
D-T          1     0.444    921.79   921.86  +0.07  (3,5) Strong gravity
D-D          1     0.476    916.92   916.99  +0.07  (3,5) Strong gravity
D-3He        2     0.888    873.48   873.55  +0.07  (3,5) Strong gravity
T-T          1     0.416    926.34   926.41  +0.07  (3,5) Strong gravity
3He-3He      4     1.664    829.72   829.79  +0.07  (3,0) Strong bare
p-6Li        3     1.278    848.13   848.19  +0.07  (3,2) Strong spinorial
p-B11        5     1.861    821.92   821.99  +0.07  (3,0) Strong bare
```

Every barrier lands on a `k = 3` class. The truncated-barrier peak coincides with the barrier tick within 0.1 ticks for all seven fuels. True resonances appear as measured offsets below the barrier. D-T at 50 keV sits `+152.22` ticks above the barrier tick on the IR side of the ruler, and p-B11 at 600 keV sits `+78.89` ticks above its barrier tick.

### 7.1.1 Barrier Radius Sensitivity

Because `V_b ∝ 1/r0`, a change of nuclear-radius prefactor shifts every barrier tick by the same amount, `Δn = −log2(r0 / 1.2) / Δ`, independent of fuel. Sweeping `r0 ∈ {1.1, 1.2, 1.3, 1.4}` fm on the seven holdout fuels (`hqvm_cgm_trestleboard_4.py`, section I) yields

```
r0 (fm)   Δn vs 1.2 (ticks)   all on k=3
1.1       −6.06               yes
1.2        0.00               yes
1.3       +5.58               yes
1.4      +10.74               yes
```

Strong-ladder placement (`k = 3`) survives the full sweep. Within `r0 ∈ [1.1, 1.3]` the tick shift stays inside the 7-tick peak-coincidence tolerance. At `r0 = 1.4` the shift exceeds that tolerance while the ladder class remains `k = 3`. Barrier-class claims are therefore robust to the usual nuclear-radius band. Peak-coincidence at the default `r0 = 1.2` is the sharper, radius-sensitive statement.

### 7.2 The Two Dials and the Cross-Section Formula

Two inclusion dials feed the coverage `θ` in the fusion cross-section. The tau-dial sets `p = p_c · T` with `T = exp(−τ)`, the Beer-Lambert form of Gamow barrier transmission,

```
τ = √(E_G / E) − √(E_G / V_b)   (E < V_b),   τ = 0 otherwise
```

Below the barrier, `τ` is the excess optical depth relative to the barrier. Above it, transmission saturates. Inverting `T = p_target` recovers the landmark energy

```
√(E_G / E) = √(E_G / V_b) − ln(p_target) ,   E_τ = E_G / [√(E_G/V_b) − ln(p_c)]²
```

so `E_τ` is the center-of-mass energy at which bare transmission equals the rank threshold `p_c`. At and above the barrier, `τ = 0` so `p_τ = p_c`. The Δ-dial sets `p_Δ = E / V_b`, with twin landmark `E_Δ = p_c · V_b`. The astrophysical S-factor convention factors the Coulomb penetration from the nuclear matrix element, writing `σ(E) = (S(E)/E) P_Gamow(E)`. The default CGM model (Model 2, dial = Δ) multiplies that baseline by the kernel coverage,

```
σ ∼ (S / E) · P_Gamow · θ(p_Δ)
```

`S` is the astrophysical factor, `P_Gamow` the Gamow penetration, and `θ(p_Δ)` the kernel coverage at inclusion `p_Δ = E / V_b`. The effective transport rank read from coverage is the exact inverse of the square-root cluster identity for `r ≥ 1`,

```
θ(r) = (2^r / |H|)² ,   r_eff = d_χ + ½ log2(θ)   (clipped to [0, d_χ], with θ ≤ 2/|Ω| mapping to r = 0)
```

The horizon size is `|H| = 64` and the chirality dimension is `d_χ = 6`. The variable `r_eff` is the coverage-inferred rank. The Gamow factor is kept separately from `θ`, so barrier penetration is counted once. Model 1 instead takes `θ` as the tunneling factor and drops the separate Gamow factor. The native model drops the Gamow factor entirely and tests whether the exact coverage `θ(p)` alone reproduces the measured cross-section. Under each of these choices, `θ(p)` remains the exact kernel coverage and supplies a lower bound on the fusion rate. Reference S-factors are the Bosch-Hale Padé fits for D-T, D-D, D-3He, and T-T, Tentori-Belloni for p-11B, Solar Fusion II for 3He-3He, and the Trojan Horse Method fit for p-6Li.

In a fusion calculation the trestleboard is used as a plug-in factor alongside R-matrix methods. Given `(Z1, Z2, A1, A2)` and a dial choice, it returns `θ(E/V_b)`, the rank landmarks `E_r = p_c(r)·V_b`, the susceptibility width proxy `Γ_struct(r)`, and the cutoff discriminant `R`. The baseline remains `σ_base = (S/E) P_Gamow`. The CGM-modulated form is `σ = σ_base · θ(E/V_b)`. Any Breit–Wigner or R-matrix resonance term is an optional overlay on that floor.

The dual dial covers all four fuels in the test set. `E_τ` is the energy where `T = exp(−τ) = p_c`. `E_Δ` is the energy where `p_Δ = p_c`.

```
fuel     E_τ(keV)  hitτ   E_Δ(keV)  hitΔ   Res(keV)  TOL   best
D-T         72.5     Y       40.3     Y      64.0    25.0  both
D-D         66.6     Y       43.2     N     100.0    40.0  τ
D-3He      212.8     Y       80.6     N     250.0   100.0  τ
p-B11      650.9     N      169.0     Y     148.0    60.0  Δ
```

Dual-dial coverage is 4/4. Light fuels sit on the τ-band. p-B11 sits on the Δ-dial. The ordering `E_τ(D-T) < E_τ(D-3He) < E_τ(p-B11)` holds. For D-T, `V_b ≈ 0.444 MeV`, `E_G ≈ 1.175 MeV`, and `E_rank ≈ 72.50 keV` lies inside the 5 to 500 keV band.

### 7.3 D-T Cross-Section Grid

Among candidate terrestrial fuels, D-T has the largest low-temperature reactivity because the reaction `D + T → ^4He (3.5 MeV) + n (14.1 MeV)` liberates 17.6 MeV, and a low-energy resonance identified in the wartime cross-section program (see Chadwick and Reed, 2024) raises its cross-section by about two orders of magnitude relative to naive D-D scaling. That resonance is why D-T is the power-fuel reference in the resonance map and why a pure geometric baseline cannot absorb the 50 keV peak.

On the D-T energy grid the Model-2 cross-section, normalized to `σ₀ = 1` at the 10 keV reference, is

```
E_cm   n       p_inc  r_eff  θ         P_Gamow   σG/σ0     σCGM/σ0
1.0    1346.7  0.0023  2.34  6.29e-03  1.30e-15  6.62e-10  1.17e-10
5.0    1234.5  0.0113  2.83  1.23e-02  2.20e-07  2.24e-02  7.72e-03
10.0   1186.2  0.0225  3.60  3.57e-02  1.96e-05  1.00e+00  1.00e+00
20.0   1137.9  0.0450  4.66  1.56e-01  4.69e-04  1.20e+01  5.22e+01
50.0   1074.0  0.1126  5.79  7.48e-01  7.85e-03  8.00e+01  1.68e+03
100.0  1025.7  0.2252  6.00  9.94e-01  3.25e-02  1.66e+02  4.61e+03
300.0   949.1  0.6755  6.00  1.00e+00  1.38e-01  2.35e+02  6.58e+03
500.0   913.5  1.0000  6.00  1.00e+00  2.16e-01  2.20e+02  6.17e+03
```

Self-checks confirm `θ(10) < 0.5`, monotone growth `θ(10) < θ(30) < θ(100)`, and `θ(10)/θ(100) < 0.5`. The pure-Gamow peak and the CGM-model peak both sit at 300.0 keV on this grid (`σG/σ0 = 234.9`, `σCGM/σ0 = 6582`). The model maximum remains unshifted by `θ` on D-T because coverage has already saturated near the Gamow peak. The analytic Gamow-only maximum `E_G/4 = 293.7` keV agrees with the grid peak to within one bin.

For p-B11 the barrier is higher (`V_b ≈ 1.861 MeV`, `E_G ≈ 22.438 MeV`). Coverage rises more slowly. At 100 keV, `θ ≈ 0.228` and `σCGM/σ0 = 1` by normalization. At 600 keV, `θ ≈ 1` and `σCGM/σ0 ≈ 5.18e+03`. The CGM enhancement relative to pure Gamow is therefore concentrated at intermediate energies where coverage is turning on.

### 7.4 The Resonance Map on the Percolation Hierarchy

The resonance map is the falsifiable fusion claim. Measured fusion resonances are placed on the percolation hierarchy. Declared landmarks per fuel are the union of six structural events, the Gamow-peak energy, and the rank-ladder twins on both dials:

```
E_span, E_full, E_spec, E_τ, E_word   from p_c(event) via τ-inversion
E_Δ                                   = p_c(rank) · V_b
E_Gamow                               = E_G / 4
E_τ_r{r}, E_Δ_r{r}                    rank-ladder twins for each predeclared p_c(r)
```

Seventeen landmarks are declared per fuel. Resonance energies `E_res` are center-of-mass peak positions taken from the literature sources cited with each fuel. For the holdout set those sources are the Bosch–Hale, Tentori, Solar Fusion II, and THM catalogs. For the map suite the literature peaks are D-T 50 keV, p-B11 600 keV, 10B-p 10 keV sub-threshold 11C, 12C-p 461 keV, 15N-p 325 keV with literature band 312–338 keV, 7Li-p 330 keV, and 6Li-p 440 keV. The map tolerance converts the literature energy window (keV) into ticks. Landmark energies on the four-fuel stress suite are

```
fuel     Res    E_span  E_full  E_spec   E_τ    E_word   E_Δ
D-T       64.0    39.6    43.0    50.1    72.5   149.6    40.3
D-D      100.0    35.4    38.6    45.3    66.6   143.8    43.2
D-3He    250.0   125.3   134.9   154.6   212.8   389.0    80.6
p-B11    148.0   421.6   448.5   501.9   650.9  1038.6   169.0
```

Each resonance tick `n(E_res)` is compared to the nearest landmark tick. A passing result requires the offset to lie within the literature tolerance and above the weakest rank threshold. Roles and placements for seven fuels with literature resonances are (CNO entries follow the solar CNO cycle rates of Adelberger et al., Solar Fusion II):

```
fuel     role        Z1Z2  E_res  landmark      off(ticks)  tol   status
D-T      power          1   50.0  E_spectrum      +0.20     6.64  PASS
p-B11    aneutronic     5  600.0  E_tau           +5.67     6.64  PASS
10B-p    aneutronic     5   10.0  E_tau_r0      +110.47    48.31  FAIL SUB
12C-p    CNO            6  461.0  E_delta_r1      +1.00     6.63  PASS
15N-p    CNO            7  325.0  E_delta_r2      +8.77     6.74  FAIL CUT
7Li-p    aneutronic     3  330.0  E_span          +2.56     7.97  PASS
6Li-p    aneutronic     3  440.0  E_tau           +3.75     6.06  PASS
```

The null model gives a single-hit probability under a log-uniform window of width `2 · tol` ticks over the sub-barrier band,

```
p_single = (2 · tol · Δ) / log2(V_b / E_band_lo)
```

`tol` is the literature tick tolerance and `E_band_lo` is the lower edge of the scanned band. For the suite, `p_single = 0.0334` (expected hits 0.23). Five of seven pass (`P(K >= 5) = 0.0000`, Bonferroni `p` by 17 events also 0.0000). Among fuels with `Z1 Z2 < 7` and no sub-threshold flag, placement is 5/5. The two non-passing fuels remain in the report to illustrate boundary conditions. 10B-p at 10 keV is a center-of-mass sub-threshold 11C resonance (`p_Δ ≈ 0.007`), and 15N-p at 325 keV is the `Z1 Z2 = 7` Rider-cutoff fuel whose resonance sits in the integer-rank gap between `r5` (0.146) and `r4` (0.219). The grammar gap coincides with Rider's terrestrial viability boundary. The calculated `E_G` matches the literature Gamow table to 0.4–0.8 percent.

### 7.5 Reactivity and the Enhancement Growth

For a thermal plasma the fusion rate density is `f = n_1 n_2 ⟨σv⟩` (with `n²/2` for like-particle fuels such as D-D). Densities `n_1`, `n_2` are number densities of the two reactants, and the reactivity `⟨σv⟩` is the velocity-averaged product of cross-section and relative speed. Meaningful `⟨σv⟩` requires temperatures of order 10–100 keV, well above ionization, so the reactants are a plasma, and the Lawson criterion then states the `nTτ` triple product needed for net power. The CGM scan approximates the relative reactivity by a trapezoid integral of the Model-2 integrand against a Maxwellian weight,

```
I(T) = ∫ P_Gamow(E) · θ(E/V_b) · exp(−E/T) dE
```

Temperature `T` is in energy units. The same integral without `θ` supplies the Gamow-only baseline. Relative reactivities for D-T, normalized at 10 keV, are

```
T_keV   ⟨σv⟩G / ⟨σv⟩G0   ⟨σv⟩CGM / ⟨σv⟩CGM0   R = CGM/G
1.0     3.57e-06          1.77e-07              0.0248
5.0     5.20e-02          2.53e-02              0.243
10.0    1.00e+00          1.00e+00              0.499
20.0    1.23e+01          1.92e+01              0.780
50.0    1.88e+02          3.57e+02              0.948
100.0   1.04e+03          2.05e+03              0.985
300.0   8.78e+03          1.76e+04              0.997
500.0   1.69e+04          3.38e+04              0.999
```

The absolute reactivity ratio `R(T) = ⟨σv⟩_CGM / ⟨σv⟩_G` rises monotonically toward 1 as `T → ∞`, because `θ` is a coverage-weighted average. Absolute peak locations sit at the grid edge (500 keV) for both Gamow and CGM, as expected for a monotone integrand. The falsifiable interior signal is the temperature of maximum `dR/dlnT`, the point where `θ` most rapidly reshapes the Maxwellian window. For D-T that temperature is 20 keV, inside the plasma band, confirming interior enhancement growth. The structural reading is that `θ` raises the low-energy tail of the fusion rate by a resonance-independent amount, on top of which a localized Breit-Wigner resonance (such as the D-T 50 keV peak) overlays as a compound-nucleus amplitude.

## 8. Quantitative Consequences for Fusion

### 8.1 Resonance Widths from Percolation Susceptibility

The susceptibility is the derivative of exact micro-reference coverage with respect to inclusion probability, evaluated by central difference on the closed form (`h = 10^{−6}`, no Monte Carlo),

```
χ(p) = [θ(p+h) − θ(p−h)] / (2h)
```

At each rank-ladder inclusion `p_c(r)` that susceptibility sets the structural width

```
Γ_struct(r) = χ_ref / χ(p_c(r))
```

with reference `χ_ref = max_r χ(p_c(r)) = χ(p_c(4)) = 8.848690`. A sharp transition (large `χ`) is a narrow structural resonance. Per-rank scaling on the D-T barrier is

```
r   p_c(r)    E = p_c·V_b (keV)   χ(p_c)   Γ_struct
1   0.292893  130.0702            0.015388   575.041
2   0.218779   97.1570            0.295041    29.991
3   0.145759   64.7298            3.035960     2.915
4   0.090795   40.3211            8.848690     1.000
```

The scaling is monotonically inverse across rungs. Rank-1 landmarks are broad. Higher-rank landmarks are narrow. The same `Γ_struct` values apply across fuels. Only the landmark energy `E = p_c(r) · V_b` changes:

```
fuel      E_r1 (keV)  E_r2     E_r3     E_r4
D-T         130.07     97.16    64.73    40.32
D-D         139.48    104.19    69.41    43.24
D-3He       260.14    194.31   129.46    80.64
T-T         121.85     91.02    60.64    37.77
3He-3He     487.39    364.06   242.55   151.09
p-6Li       374.29    279.58   186.27   116.03
p-B11       545.09    407.16   271.27   168.98
```

For a beam-target or colliding-beam system with a controlled energy profile, `χ` sets the required energy spread `ΔE / E` to lock a given rank closure. A Maxwellian distribution clips the narrow landmarks weakly. A monoenergetic beam can force them.

### 8.2 The CGM Surrogate on Predictive Holdout

The CGM cross-section surrogate is tested on the reference S-factor tables in `data/catalogs/fusion/` with a single scale degree of freedom `C`:

```
σ_CGM(E) = C · P_Gamow(E) · θ(E / V_b) / E
```

Calibration uses even CSV indices and fits `C` by least squares against the reference cross-section `σ_ref = S_ref · P_Gamow / E`, namely `C = Σ σ_ref · σ_raw / Σ σ_raw²` with `σ_raw` the unscaled CGM shape. Holdout uses odd indices. Per-fuel holdout metrics are

```
fuel      n_cal  n_hold  C            RMSE(log10)  Pearson r
D-T          48     48   6.981e+06    0.5777       -0.1175
D-D          48     48   1.295e+07    0.2308        0.9384
D-3He        48     48   1.099e+05    0.4526        0.9985
T-T          48     48   8.610e+04    0.2283        0.9888
3He-3He      48     48   6.392e+03    0.8175        0.9955
p-6Li        48     48   2.958e+03    0.7425        0.9904
p-B11        61     60   3.205e+05    0.7906        0.9710
```

Pooled RMSE in log10 is 0.5962. Pooled mean Pearson `r` is 0.8236. Non-resonant fuels give `r` above 0.93. The D-T holdout Pearson `r` is minus 0.12. That result is consistent with the role of `θ(p)` as the direct, non-resonant topological baseline. The Breit-Wigner peak near 50 keV is a localized compound-nucleus overlay absent from the baseline. The holdout therefore measures baseline shape. The decomposition used here is a geometric floor (`θ(p)` from the barrier tick, a resonance-independent lower bound on `⟨σv⟩`) together with a resonance boost at specific energies. When the geometric floor fails the Lawson criterion (Lawson, 1957) for a fuel, resonance structure leaves viability unrestored.

### 8.3 The Rider Cutoff as an Internal Discriminant

The Rider cutoff (Rider, LLNL High Energy Density Science seminar, 19 January 2023) marks `Z1 Z2 >= 7` as the Coulomb barrier too high for terrestrial fusion, with `Z1 Z2 >= 8` absolute, and notes that p-11B already has a bremsstrahlung-to-fusion power ratio of 1.19 in equilibrium plasma. That fuel is the canonical aneutronic candidate (`p + ^11B → 3 ^4He + 8.7 MeV`), but advanced fuels pay a radiation-loss penalty that grows with `Z` of the non-hydrogenic reactant. The CGM supplies an internal discriminant from barrier placement alone:

```
R(Z1 Z2) = n(V_b(Z1 Z2)) - n_cut
```

with `n_cut = n(V_b)` at `Z1 Z2 = 7` (15N-p) equal to 828.86 ticks. Per-fuel values are

```
fuel     Z1Z2   n(V_b)   R        below cutoff
D-T         1   921.79  +92.93    True
D-D         1   916.92  +88.06    True
D-3He       2   873.48  +44.62    True
p-6Li       3   848.13  +19.26    True
p-B11       5   821.92   -6.94    True (anomaly)
10B-p       5   843.26  +14.39    True
12C-p       6   834.61   +5.74    True
15N-p       7   828.86   +0.00    False
```

`R` is positive on the accessible side and negative above the cutoff. Strict sign separation fails only on p-B11, where `R ≈ −6.94` despite `Z1 Z2 = 5`, because the barrier tick sits below the `Z1 Z2 = 7` reference. This discriminant correlates barrier placement with the cutoff anchor without computing bremsstrahlung or deriving Rider's `P_brem/P_fus` ratio. The geometric counterpart of Rider's radiation-loss marginality for p-B11 is that the barrier placement itself sits on the wrong side of the cutoff tick. The classification used here is that `R <= 0` reports a geometry-restricted channel relative to the cutoff reference.

### 8.4 Sparse-Data Prediction Targets

For each holdout fuel, the untested band is `[first CSV energy, p_c(1) * V_b]`. Rank-1 landmarks and untested widths are

```
fuel      first_CSV (keV)  landmark (keV)  untested width (keV)
D-T              10.0           130.07              120.1
D-D              10.0           139.48              129.5
D-3He            10.0           260.14              250.1
T-T              10.0           121.85              111.8
3He-3He          10.0           487.39              477.4
p-6Li            10.0           374.29              364.3
p-B11            10.0           545.09              535.1
```

These are targeted beam energies for structural rank transitions where standard S-factor tables are silent below the landmark.

## Appendix A. Design Hypotheses for Three Fusion Domains

This appendix is speculative. None of its claims carries a formal validation gate in this analysis. It records which coordinate each experimental route acts on once the core modules have fixed `p = E/V_b`, `θ(p)`, and `p_c(r)`.

```
domain              coordinate changed              CGM lever
compact hot         raise E (hence p)               place power on broad Γ_struct bands
muon-catalyzed      ρ-dress (orbit shrink)          Gate F / carrier-trace scaling
lattice / LENR      supply missing generators       local rank-6 completion at defects
```

Kinetic heating raises `p` until thresholds are crossed stochastically. The other two routes change geometry while leaving the kinetic coordinate fixed.

### A.1 Compact Hot Fusion

The standard terrestrial routes, namely magnetic confinement (tokamak, stellarator) and inertial confinement (laser or beam drivers), raise kinetic energy until the plasma triple product approaches the Lawson threshold. Under the CGM reading, heating raises `p` until the plasma distribution crosses rank thresholds. Shaping RF or beam energy deposition onto the broad `Γ_struct` landmarks (rank-1 and rank-2) instead of uniform heating targets the broad `χ` bands and can reduce auxiliary power for a given `θ`. The route remains kinetic, and the change is energy placement on the ruler.

### A.2 Muon-Catalyzed Fusion

Muon-catalyzed fusion proceeds at ordinary temperatures because the muon mass shrinks the Bohr orbit by about 207 times, so nuclei sit closer without MeV thermal `E`. Net energy production has remained unsuccessful. Muon production is costly, the muon lifetime is 2.2 μs, and sticking of the muon to the daughter alpha terminates the catalysis chain (Jones, 1986). Under the CGM reading, the muon is a forced `ρ`-dress, a spinorial mass scaling that shifts the bipartite carrier toward complement-horizon closure without adding kinetic `E`. Sticking is the muon trapped in the daughter Z2 holonomy after the Gate F word closes. An open question is whether the physical muon is required or only its operator signature. If an electromagnetic drive can match the Δ-step or carrier-trace scaling of the discrete pion parallel, including the forbidden modes via `C(3) = 52/25`, the geometric close would occur without particle production or sticking. That possibility remains a hypothesis outside the scope of formal validation in this analysis.

### A.3 The Lattice Path

A standard objection to lattice fusion is that lattice thermal energy cannot overcome `V_b`. The same objection appears in beam-target fusion, where fusion cross-sections are many orders of magnitude below Coulomb scattering, so most ions radiate or ionize before fusing. Under the CGM reading, `E` is one coordinate of `p = E / V_b`, and rank-6 generator completion on `GF(2)^{d_χ}` with `d_χ = 6` is the fusion transition condition. A metal lattice (Pd, Ni, and others) can supply fixed, pre-loaded chirality-transport bytes at defect sites (dislocations, grain boundaries, vacancies), so the local generator set reaches rank 6 without raising the kinetic energy through the barrier. The implied consequences are rank completion as a slow holonomic process (`θ → 1` locally), replication failures from uncontrolled metallurgy when one defect geometry completes rank 6 and a nearby sample remains at rank 5, and a spectrum that may select aneutronic or low-energy alpha branches fixed by lattice holonomy. These implications require pathway enumeration against EXFOR and the contested LENR literature. These implications describe geometric completion of the generator set at defect sites.

## Appendix B. External Data Provenance

Frozen nuclear-structure snapshots reside in `data/catalogs/ensdf/` with provenance in `data/catalogs/ensdf/SOURCE.txt`. Ground-state spins, parities, alpha and beta parents, Q-values, and half-lives are read from `iaea_livechart_ground_states.csv` (IAEA Nuclear Data Section LiveChart API, underlying evaluations ENSDF). The eV-band isomer census is the filtered table `ensdf_ev_band_levels.csv` (0 < E ≤ 200 eV from 214 actinide level files, Z = 88–98, A = 220–250), and first-excited actinide energies are in `ensdf_first_excited_actinides.csv`. The API endpoint is https://nds.iaea.org/relnsd/v1/data.

The primary optical-isomer comparison for Th-229m uses Zhang et al., Nature 633, 63–70 (2024), DOI 10.1038/s41586-024-07839-6, at 8.3557335(8) eV in CaF2 (local copy and provenance in `docs/references/SOURCE_Th229m.txt`). That measured value supersedes the ENSDF Adopted listing for Th-229, which still records approximately 7.6 eV.

The eV-band census also includes U-235m near 76 eV (Ponce et al., Phys. Rev. C 97, 054310, 2018; Shigekawa et al., arXiv:2603.01699, 2026), which lies outside the predicted optical-isomer window of this analysis (local copies in `docs/references/SOURCE_U235m.txt`).

Reference astrophysical S-factors for holdout tests reside in `data/catalogs/fusion/` with provenance in `data/catalogs/fusion/SOURCE.txt`. D-T, D-D, D-3He, and T-T use the Bosch–Hale Padé fits (Nucl. Fusion 32, 611, 1992). p-11B uses the Tentori–Belloni piecewise fit (Nucl. Fusion 63, 086001, 2023). 3He-3He uses the Solar Fusion II quadratic (Adelberger et al., Rev. Mod. Phys. 83, 195, 2011). p-6Li uses the Trojan Horse Method quadratic fit recorded in that catalog.

The electroweak anchor `v` and the W and Z masses used in the mass-ratio recovery of the ruler unit follow the PDG 2024 review (Navas et al., Phys. Rev. D 110, 030001, 2024). The deuteron binding energy used for the strong-scale check is 2.2240 MeV (PDG few-nucleon summary).

The terrestrial fusion viability boundary is the Rider cutoff `Z1 Z2 ≥ 7` (with `Z1 Z2 ≥ 8` absolute) together with the p-11B bremsstrahlung-to-fusion power ratio 1.19, both taken from Rider, LLNL High Energy Density Science seminar, 19 January 2023, "Is There a Better Route to Fusion?" (slides: https://heds-center.llnl.gov/sites/heds_center/files/2023-03/01-19-23_slides_-_rider_.pdf).

## References

1. B. Korompilias, *Common Governance Model: Mathematical Physics Framework*, Zenodo (2025), https://doi.org/10.5281/zenodo.17521384.
2. Particle Data Group, S. Navas et al., *Review of Particle Physics*, Phys. Rev. D 110, 030001 (2024), https://doi.org/10.1103/PhysRevD.110.030001.
3. C. Zhang et al., *Frequency ratio of the 229mTh nuclear isomeric transition and the 87Sr atomic clock*, Nature 633, 63-70 (2024), https://doi.org/10.1038/s41586-024-07839-6; arXiv:2406.18719.
4. F. Ponce, E. Swanberg, J. Burke, R. Henderson, and S. Friedrich, *Accurate measurement of the first excited nuclear state in 235U*, Phys. Rev. C 97, 054310 (2018), https://doi.org/10.1103/PhysRevC.97.054310.
5. Y. Shigekawa et al., *Chemical effects on nuclear decay of 235U isomer in the uranyl form*, arXiv:2603.01699 (2026).
6. IAEA Nuclear Data Section, LiveChart of Nuclides Data Download API, https://nds.iaea.org/relnsd/v1/data; underlying evaluations: Evaluated Nuclear Structure Data File (ENSDF).
7. Laboratory National Henri Becquerel (LNHB), Recommended data for 3H beta decay.
8. H.-S. Bosch and G.M. Hale, *Improved formulas for fusion cross-sections and thermal reactivities*, Nucl. Fusion 32, 611 (1992), https://doi.org/10.1088/0029-5515/32/4/I07.
9. A. Tentori and F. Belloni, *Revisiting p-11B fusion cross section and reactivity, and their analytic approximations*, Nucl. Fusion 63, 086001 (2023), https://doi.org/10.1088/1741-4326/acda4b.
10. E.G. Adelberger et al., *Solar fusion cross sections. II. The pp chain and CNO cycles*, Rev. Mod. Phys. 83, 195 (2011), https://doi.org/10.1103/RevModPhys.83.195.
11. A. Tumino, C. Spitaleri, et al., *Indirect study of the astrophysically relevant 6Li(p, alpha)3He reaction by means of the Trojan Horse Method*, Prog. Theor. Phys. Suppl. 154, 341 (2004), https://doi.org/10.1143/ptps.154.341.
12. T.H. Rider, *Is There a Better Route to Fusion?*, LLNL High Energy Density Science Seminar, 19 January 2023, https://heds-center.llnl.gov/sites/heds_center/files/2023-03/01-19-23_slides_-_rider_.pdf.
13. G. Gamow, *Zur Quantentheorie des Atomkernes*, Z. Phys. 51, 204–212 (1928), https://doi.org/10.1007/BF01343196.
14. R. d'E. Atkinson and F.G. Houtermans, *Zur Frage der Aufbaumöglichkeit der Elemente in Sternen*, Z. Phys. 54, 656–665 (1929), https://doi.org/10.1007/BF01341595.
15. J.D. Lawson, *Some Criteria for a Power Producing Thermonuclear Reactor*, Proc. Phys. Soc. B 70, 6–10 (1957), https://doi.org/10.1088/0370-1301/70/1/303.
16. S.E. Jones, *Muon-Catalysed Fusion Revisited*, Nature 321, 127–133 (1986), https://doi.org/10.1038/321127a0.
17. M.B. Chadwick and B.C. Reed, *Introduction to Special Issue on the Early History of Nuclear Fusion*, Fusion Sci. Technol. 80, S1 (2024), https://doi.org/10.1080/15361055.2024.2346868.
18. Companion analyses: `docs/Findings/Analysis_Compact_Geometry.md`, `docs/Findings/Analysis_hQVM_Percolation.md`, `docs/Findings/Analysis_hQVM_Cohomology.md`, `docs/Findings/Analysis_Gravity_Note.md`, `docs/Findings/Analysis_Hilbert_Space_Representation.md`, and `docs/CGM_Logic.md`. Kernel specification layer: `docs/Gyroscopic_Computational_Theory/hQVM_Specs_Formalism.md`, `docs/Gyroscopic_Computational_Theory/hQVM_Features_Report.md`, `docs/Gyroscopic_Computational_Theory/hQVM_QuBEC_Theory.md`, and `docs/Gyroscopic_Computational_Theory/hQVM_SDK_Quantum_Computing.md`.
19. Data catalogs: `data/catalogs/ensdf/` and `data/catalogs/fusion/` (SOURCE files in each directory); local isomer PDFs in `docs/references/`.
