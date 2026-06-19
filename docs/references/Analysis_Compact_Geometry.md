# Compact Geometry: Spectral Algebra of the Electroweak Mass Spectrum

## 1. Scope, Claims, and Dependency Map

### 1.1 Scope

The Common Governance Model (CGM) derives the structure of physical space and its fundamental constants from five logical conditions on coherent recursive measurement. Within this framework, the finite kernel is a discrete algebraic system with 4,096 reachable states, organised into seven shells by a binomial distribution and carrying a self-dual [12,6,2] binary code. The kernel supplies exact combinatorial and spectral data with no freely adjustable parameters.

This report uses the finite kernel as the sole algebraic input for a mass-coordinate law covering the four principal electroweak observables: the top quark, Higgs boson, Z boson, and W boson. The law expresses each mass as a spectral expansion in powers of the aperture parameter Delta, which measures the fractional non-closure of the CGM depth-four cycle relative to the observational aperture scale. Delta is fixed independently of the electroweak masses by the CGM geometric invariants. The expansion extends from Delta^1 through Delta^5, and every coefficient is a fixed rational number drawn from the kernel's discrete grammar of shell multiplicities, horizon cardinalities, code weights, and gyroscopic stage flags. At fifth order, the maximum tick error across the four channels is 6.15 × 10⁻⁹, and the W/Z ratio recovers the independently defined aperture Delta to 8.34 × 10⁻¹⁰.

The analysis is organised in three claim layers:

1. **Exact finite-kernel facts.** The 4,096-state reachable manifold, the dual 64-state horizons, the seven-shell binomial spectrum, the self-dual [12,6,2] code chart, and the reduced shell quantities C1 = 6, C2 = 15, C3 = 20, M_shell = 192. These are combinatorial consequences of the kernel definition and require no physical input.

2. **Electroweak coordinate law.** The projection of the finite spectrum into four physical mass channels. Once Delta and the channel assignment (which kernel channel corresponds to top, Higgs, Z, W) are supplied, the law coefficients are fully determined by discrete kernel data: shell multiplicities, horizon size, K4 stage flags, trace-free gyroscopic charges, and code-curvature terms.

3. **External and imported hypothesis channels.** Neighbouring layers supply additional closure machinery: the horizon-wrap rule, the 0xAA source-traceability theorem, and the lifted K6 normalisation. These are explicitly marked where they enter and are treated as imported inputs rather than derived results.

Observed inputs enter as boundary data only:
* Top mass, Higgs boson mass, and Z/W masses from PDG.
* CODATA/NIST values for electromagnetic and related constants.

Input convention ledger:

| Symbol | Value | Convention |
| ------ | -----: | ---------- |
| `v` | 246.22 GeV | PDG electroweak scale convention used in this report |
| `m_t` | 172.760 GeV | PDG top mass convention (electroweak default) |
| `m_H` | 125.100 GeV | PDG central value |
| `m_Z` | 91.187600 GeV | PDG 2024 review |
| `m_W` | 80.379000 GeV | PDG 2024 review |

Lambda_QCD = 0.2000 GeV is a conventional scale parameter placed on the ruler.

### 1.2 Dependency Status of Major Claims

| Level | Claim | Status |
| ----- | ----- | ------ |
| 1 | Finite kernel: `|Omega|` = 4096, dual `|H|` = 64-state horizons, seven-shell binomial structure, self-dual [12,6,2] enumerator | Derived in this report |
| 1 | Shell-code quotient and reduced quantities C1=6, C2=15, C3=20, M_shell=192 | Derived in this report |
| 2 | Electroweak law: `L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5` | Derived in this report |
| 2 | Numerical electroweak audits and 24-bit obstruction / 32-bit lift necessity | Derived in this report |
| 2 | Horizon-wrap theorem, 0xAA source-traceability, lifted K6 normalisation | Imported from neighbouring layers |
| 2 | PDG electroweak masses, conventional QCD scale | External inputs |
| 3 | CKM ansatz, atomic spectroscopy, antihydrogen, redshift, quark selectors, lepton carrier closures | Deferred hypothesis channels |

## 2. Finite Kernel and Operator Algebra

This section establishes the exact combinatorial and spectral structure of the finite kernel. All results here are consequences of the kernel definition alone and are independent of the electroweak mass data.

### 2.1 Reachable Manifold and Dual Horizons

The CGM finite kernel is a deterministic algebraic system whose state is a 24-bit register formed from two conjugate 12-bit gyrophases, denoted A and B. The kernel evolves by applying bytes (8-bit instruction units) under a fixed transition law. The reachable manifold Omega is the set of states accessible from the rest state under this byte transition law.

Exhaustive enumeration gives:

| Quantity          |                         Value |
| ----------------- | ----------------------------: |
| Reachable states  |                          4096 |
| Form              |            2^12 = 16^3 = 64^2 = 4096 |
| Component density | popcount(A) = popcount(B) = 6 |
| Density product   |            d(A) x d(B) = 0.25 |

The constant component density is a structural constraint of the reachable manifold. It follows from the pair-diagonal mask structure and the self-dual [12,6,2] code chart.

Exhaustive byte transitions are checked exactly:

```text
4096 states x 256 bytes = 1,048,576 operations
active swap failures            = 0
passive commit failures         = 0
complement swap fraction        = 0.500000
complement commit fraction      = 0.500000
equality horizon hit fraction   = 0.015625
complement horizon hit fraction = 0.015625
```

Omega contains two disjoint extremal 64-state subsets.

| Horizon            | Condition     | Chirality |
| ------------------ | ------------- | --------- |
| Equality horizon   | A = B         | zero      |
| Complement horizon | A = B xor 0xFFF | maximal   |

The rest state lies on the complement horizon. The two horizons form a 128-state boundary of Omega. The remaining 3968 states form the relational bulk. Each horizon has cardinality 64, and the bulk manifold satisfies the holographic counting identity:

```text
|H|^2 = |Omega| = 64^2 = 4096.
```

### 2.2 Complementarity Invariant

For every state s in Omega, the following exact invariant holds:

```text
horizon_distance(s) + ab_distance(s) = 12.
```

This is an exact invariant of the compact state. No state can occupy both horizon conditions simultaneously.

### 2.3 Shell Distribution and Code Chart

Omega partitions into seven shells according to ab_distance.

| Shell | ab_distance | Population | Fraction | Characterisation   |
| ----: | ----------: | ---------: | -------: | ------------------ |
|     0 |           0 |         64 | 0.015625 | equality horizon   |
|     1 |           2 |        384 | 0.093750 | near unity         |
|     2 |           4 |        960 | 0.234375 | intermediate       |
|     3 |           6 |       1280 | 0.312500 | equatorial maximum |
|     4 |           8 |        960 | 0.234375 | intermediate       |
|     5 |          10 |        384 | 0.093750 | near opposition    |
|     6 |          12 |         64 | 0.015625 | complement horizon |

Shell index k is used globally as shell_k with ab_distance = 2k. The companion gravity analysis reports the equivalent coordinate arch_shell = 6 - k (arch_shell 0 = complement, arch_shell 6 = equality).

The population law is:

```text
|Shell_k| = C(6,k) x 64.
```

The unique maximal shell is Shell 3, where ab_distance = 6.

The kernel masks form a self-dual binary [12,6,2] code. Its weight enumerator has weights 0, 2, 4, 6, 8, 10, 12 with counts 1, 6, 15, 20, 15, 6, 1. Multiplication by the horizon size 64 gives the shell populations. The code chart and the shell chart are the same binomial structure viewed through coding and state-space coordinates.

### 2.4 Byte Projection Structure

The 256-byte alphabet projects to Omega through a two-stage projection and fibre structure:

```text
256 bytes -> 128 Omega-maps -> 64 q-classes.
```

The full byte-to-q projection has uniform 4-to-1 fibres. The q-class structure is the 6-bit payload structure of the byte algebra.

The 1/64 structural law is a consequence of this fibre structure. Three independently derived kernel fractions coincide:

| Property                               |     Fraction |
| -------------------------------------- | -----------: |
| Horizon-maintaining bytes              | 4/256 = 1/64 |
| Byte commutativity rate                | 4/256 = 1/64 |
| q-fibre size relative to byte alphabet | 4/256 = 1/64 |

The common fraction is 1/64 = 2^-6, reflecting a reciprocal 64 = 2^6 six-bit payload scale.

From the complement horizon, a boundary-to-bulk projection sends the one-byte fanout uniformly to Omega:

```text
64 horizon states x 256 bytes = 16384 operations = 4 x 4096 target states.
```

Every state in Omega is reached with multiplicity 4. This gives an exact finite boundary-to-bulk dictionary.

### 2.5 Shell Number Operator and Spectral Triple

The spectral structure of the canonical operator layer is defined by:

```text
A = functions on Omega
H = l^2(Omega)
T_x : psi(s) -> psi(T_x(s))     (byte transition permutation on Omega)
K(x,y) = T_x T_y T_x^-1 T_y^-1  (commutator defect)
D_shell = sum k=0..6 k * P_k     (shell number operator on Omega)
```

Here P_k projects onto shell k.

D_shell is diagonalized by the Krawtchouk-Walsh basis on GF(2)^6, in which shell projectors and spectral projectors are dual idempotent bases of the same finite Hamming scheme. Its spectral multiplicities are:

```text
Full Omega spectrum:
dim P_k = 64 * C(6,k),   k = 0..6
Tr(D_shell) = 64 * 192 = 12288

Reduced code-shell quotient:
dim Pi_k = C(6,k),   k = 0..6
Tr(D_code) = 192
```

The electroweak coefficient alphabet reuses the reduced code-shell spectrum:

```text
{C1, C2, C3} = {C(6,1), C(6,2), C(6,3)} = {6, 15, 20}
M_shell = M1 = sum k=0..6 k * C(6,k) = 192 = Tr(D_code).
```

The electroweak coefficients are eigenvalue multiplicities of the shell number operator.

The canonical non-commutative geometry object associated with the kernel is the finite triple:

```text
A = algebra generated by {T_b : b in 0..255}
H = l^2(Omega)
D = D_shell = sum k=0..6 k * P_k.
```

Here A is a finite matrix algebra of byte-transition permutations on Omega, H is the 4096-dimensional Hilbert space of square-summable functions on Omega, and D_shell is the shell number operator. The commutator layer is generated by:

```text
[D_shell, T_b]
K(x,y) = T_x T_y T_x^-1 T_y^-1.
```

Because Omega is finite, boundedness of commutators is automatic at the matrix level. 

Using the A/B swap as the candidate grading gamma and the 0xAA S-gate as the candidate real structure J, the relations gamma^2 = I, J^2 = I, J D_shell = D_shell J, and J gamma = gamma J are verified. In the 24-bit representation, gamma commutes with D_shell instead of anticommuting. This is a structural consequence of the projection from the 32-bit register algebra to the 24-bit spatial shadow, which collapses the four family phases into two spatial actions. The shadow representation carries only the commutative grading; the anticommutative grading is recovered in the lifted 32-bit sector.

The kernel operator group from Omega has the exact finite algebraic form:

```text
G = (GF(2)^6 x GF(2)^6) semidirect C2
C2 action: swaps the two GF(2)^6 coordinates
translation subgroup size          = 4096
parity sector size                = 2
operator group size               = 8192
derived subgroup G'               = 64
centre Z(G)                        = 64
G' = Z(G)                          = diagonal GF(2)^6
abelian shadow G/G'                 = 128
two-step uniformisation             = exact translation surjection
```

Define the chirality-flow candidate operator:

```text
D_flow = popcount(A) - popcount(B).
```

The executable confirms gamma anticommutation with the flow operator:

```text
D_flow gamma + gamma D_flow = 0.
```

`D_shell` is the primary shell-number operator used by the electroweak law. `D_flow` is the chirality-flow candidate for finite Dirac-like grading tests in lifted settings. The lifted K4-gauge sector recovers the required graded first-order behavior. The construction of a full lifted spectral triple beyond the K4-gauge sector is outside the scope of this report.

## 3. Aperture Delta and the Compact Ruler

### 3.1 Continuous and Discrete Aperture

The CGM aperture is defined from the BU monodromy defect and the observational aperture scale:

| Quantity               | Expression       |          Value |
| ---------------------- | ---------------- | -------------: |
| Observational aperture | m_a = 1/(2 sqrt(2π)) | 0.199471140201 |
| BU monodromy           | d_BU             | 0.195342176580 |
| Closure ratio          | rho = d_BU/m_a       | 0.979300446087 |
| Aperture gap           | Delta = 1 - rho      | 0.020699553913 |

Delta measures the fractional non-closure of the BU cycle relative to the observational aperture scale.

| Aperture            |                  Value | Origin                        |
| ------------------- | ---------------------: | ----------------------------- |
| Bare byte aperture  | 5/256 = 0.019531250000 | 8-bit dyadic byte grid        |
| Continuous aperture |     Delta = 0.020699553913 | BU monodromy gap              |
| Depth-4 aperture    |  1/48 = 0.020833333333 | 4-byte x 12-bit closure frame |

The ordering is exact:

```text
5/256 < Delta < 1/48.
```

The depth-4 product is 48*Delta = 0.993578587835. The residual from exact depth-4 closure defines epsilon:

```text
epsilon = 1/Delta - 48 = 0.310219833351.
```

The second conversion depth is:

```text
eta = m_a - d_BU = 0.004128963621.
```

This conversion-depth quantity serves a bookkeeping role and lies outside the electroweak mass-coordinate hierarchy.

The aperture frame is:

```text
APERTURE_FRAME = 3 x |K4|^2 = 48.
```

The kernel boundary projector is:

```text
P = 1 - 1/APERTURE_FRAME = 47/48 = 0.979166666667.
```

The ratio between the depth-4 aperture and the turn quantisation is (1/48)/(1/32) = 2/3: two chiral gyrophase layers over three spatial axes.

### 3.2 Delta Self-Consistency

Three-factor reconstruction.

The continuous aperture Delta is reconstructed from the bare byte aperture by:

```text
Delta = (5/256) x 2^(1/12) x (1 + (sqrt(6)/π)Delta^2).
```

The factors are:

| Factor       | Meaning                                       |
| ------------ | --------------------------------------------- |
| 5/256        | bare dyadic byte aperture                     |
| 2^(1/12)     | UNA rotational dressing per degree of freedom |
| 1 + (sqrt(6)/π)Delta^2 | second-order 6-DoF chirality correction       |

Numerical closure. Evaluation at D^2 order gives 0.020699551515. The CGM value is Delta = 0.020699553913. The D^2 residual is approximately 2.4 x 10^-9. Evaluation at D^3 order gives 0.020699553957, with a residual of approximately 4.4 x 10^-11. The D^3 formula has a genuine fixed point at this value (self-consistency residual below 10^-15). The D^3 self-consistency closes two orders of magnitude more tightly than the D^2 evaluation.

Fixed-point behaviour. Iteration from the bare seed 5/256 converges rapidly:

| Iteration |     Delta estimate | Error from Delta |
| --------: | -------------: | -----------: |
|         1 | 0.020698793188 |   7.6 x 10^-7 |
|         2 | 0.020699551006 |   2.9 x 10^-9 |
|         3 | 0.020699551513 |   2.4 x 10^-9 |

The D^2 fixed point is stable at the displayed precision. Extending the reconstruction to D^3 order yields a fixed point at 0.020699553957, with error from CGM Delta of approximately 4.4 x 10^-11.

### 3.3 Delta Ruler Coordinates

For an energy anchor `E0` and observable energy `E`, the Delta-ruler separation is:

```text
n = log2(E0/E)/Delta.
```

For electroweak mass coordinates, `E0` is the electroweak scale `v = 246.22 GeV`. Pairwise ratios are anchor-independent.

The Refractive Depth interpretation is:

```text
tau = n * Delta * ln2.
```

Compact observations from this ruler:

```text
alpha_geometric = d_BU^4/m_a = 7.299683x10^-3
```

`d_BU^4` means `d_BU` raised to the fourth power.

The base geometric derivation is in the CGM Constants analysis, and the CGM unit framework that fixes their physical scaling is in the CGM Units analysis.

Strong-sector scale check:

```text
n_QCD from EW/0.2 = 495.940
n_QCD from top anchor = 495.940
BU-strong offset = -0.000
```

The two independent estimates agree at reported precision.

## 4. Electroweak Mass-Coordinate Law

This section presents the central structural result: the electroweak mass coordinates are given by a carrier-trace polynomial on the shell-path ladder. Each power of Delta extracts a distinct layer of the kernel's finite spectral structure.

### 4.1 Carrier-Trace Polynomial

The electroweak mass-coordinate law is a carrier-trace polynomial on the shell-path ladder. The shell path for each channel fixes a, b, c from horizon and projector data. The K4 flags fix p, q from gyrotriangle closure. The code curvature fixes r5. The wavefunction theorems of the kernel supply the structural reasons.

| Theorem / structure | Coefficient |
| ------------------- | ----------- |
| T_shells (shell path, horizon) | a_i, b_i |
| T_carrier_traces (stage projections) | c_i |
| Gyrotriangle closure | p_i |
| K4 closure (trace-free edge increments) | q_i |
| Code curvature formula | r5_i |
| K4/32-bit normalisation | 1/sqrt(5) in Delta^3 term |

The carrier-trace polynomial admits an optical reading where the shell structure is the eigen-opacity spectrum and the mass coordinate is a transmission coefficient. This optical interpretation is a physical consequence of the algebra, secondary to the derivational structure.

### 4.2 Coefficient Derivation

Declared electroweak grammar:

```text
C1 = 6, C2 = 15, C3 = 20, |H| = 64, M_shell = 192, P = 47/48, Q = 1/4.
```

Allowed electroweak orders are Delta through Delta^5; Delta^6 is the complement-horizon residual layer.

Boundary and density projectors:

```text
P = 1 - 1/APERTURE_FRAME = 1 - 1/48 = 47/48,
Q = d_A x d_B = (6/12)^2 = 1/4.
```

The K4 gyroscopic multiplication satisfies the gyrotriangle angle sum:

```text
delta = π/2 - (π/4 + π/4 + π/4) = 0.
```

This closure forces the stage ratio constraint:

```text
ONA/CS = UNA^2 = 1/2.
```

The gyrotriangle closure is the structural source of the Delta^3 and Delta^4 corrections.

K4 edge increments and trace-free charges. The K4 stage structure has three binary flags per channel: base (gyration), rot (UNA rotational), and bal (ONA balance). Each flag contributes a code-geometric edge increment:

```text
p edges: -C1/2 = -3,  +C1/4 = +3/2,  +4g = +2
q edges: 0,  -4g = -2,  -2g = -1
```

where g = 1/2 is the gyro factor from ONA/CS = UNA^2 = 1/2.

The stage flags for the four electroweak channels are:

| Channel | base | rot | bal |
| ------- | ---- | --- | --- |
| Top     | 0    | 0   | 0   |
| Higgs   | 1    | 0   | 0   |
| Z       | 1    | 1   | 0   |
| W       | 1    | 1   | 1   |

Using the trace-free conditions, the base offsets are fixed:

```text
4*p0 + sum(p increments over channels) = 0
4*p0 + (0) + (-3) + (-3/2) + (1/2) = 0
4*p0 - 4 = 0
p0 = 1

4*q0 + sum(q increments over channels) = 0
4*q0 + (0) + (0) + (-2) + (-3) = 0
4*q0 - 5 = 0
q0 = 5/4 = 1.25
```

This explains the constants in the table below and makes the trace-free conditions explicit.

Computing p_i and q_i from the edge increments:

| Channel | p_i  | q_i   |
| ------- | ---- | ----- |
| Top     | 1.0  | 1.25  |
| Higgs   | -2.0 | 1.25  |
| Z       | -0.5 | -0.75 |
| W       | 1.5  | -1.75 |

Trace-free verification:

```text
S p_i = 1.0 + (-2.0) + (-0.5) + 1.5 = 0.0
S q_i = 1.25 + 1.25 + (-0.75) + (-1.75) = 0.0
```

The trace-free property is a structural consequence of the gyrotriangle closure: the four channels span the complete K4 flag space, so the edge increments must sum to zero.

The complete electroweak mass law is:

```text
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```

| Order | Source | Algebraic origin |
| ----- | ------ | ---------------- |
| Delta^1 | Code enumerator + aperture frame | (1+z^2)^6 -> 6, 15, 20, 64, 192 |
| Delta^2 | Stage projections of code weights | -M_shell/8, +C1/4, -C3/2, +Q |
| Delta^3 | Gyroscopic phase (p-charge) | K4 edge increments, trace-free |
| Delta^4 | Gyroscopic closure (q-charge) | K4 edge increments, trace-free |
| Delta^5 | Code curvature (r5-charge) | C1, C2, \|H\|, stage flags |

| Channel | a_i | b_i | c_i | p_i | q_i | r5_i | Stage |
| ------- | ---: | ---: | ---: | ---: | ---: | ---: | ----- |
| Top | 73 | -1 | +1/4 | 1.0 | 1.25 | -4.500 | CS |
| Higgs | 96 | -1 | -24 | -2.0 | 1.25 | 2.375 | CS-UNA |
| Z | 117 | -47/48 | -22.5 | -0.5 | -0.75 | -4.500 | UNA |
| W | 126 | -47/48 | -32.5 | 1.5 | -1.75 | -2.625 | UNA-ONA |

The Delta^2 decomposition:

```text
base gyration:  -M_shell/8 = -24
UNA rotational: +C1/4  = +1.5
ONA balance:    -C3/2  = -10
```

The Delta^3 amplitude is:

```text
lambda0 = Delta/sqrt(5) = 0.009257121931.
```

The factor 1/sqrt(5) sets the K4 third-order normalisation scale used in the channel term.

The channel constant offsets are:

```text
b_top   = -1
b_higgs = -1
b_Z     = -P = -47/48
b_W     = -P = -47/48
```

Thus `b_i` is selected from one matter branch (`-1`) and two gauge branches (`-P = -47/48`).

In this report, `1/sqrt(5)` is treated as a normalisation constant imported from the K4 gyro-normalisation convention.

The Delta^5 coefficients are exact rationals from the code algebra:

```text
r5_i = -(C2-C1)/2 + (|H|-(C2-C1))/8 * (base-rot) + C2/8 * bal
```

With |H| = 64, C1 = 6, C2 = 15:

| Channel | r5_i   |
| ------- | ------ |
| Top     | -4.500 |
| Higgs   | 2.375  |
| Z       | -4.500 |
| W       | -2.625 |

These are code-valued. No fitting is involved.

### 4.3 Residuals and Delta^6 Boundary

The five-order expansion residuals are:

| Channel | p    | q    | r5     | n_err Delta^2       | n_err Delta^3       | n_err Delta^4       | n_err Delta^5       | L_err/Delta^6 |
| ------- | ---- | ---- | ------ | -------------- | -------------- | -------------- | -------------- | -------- |
| Top     | 1.0  | 1.25 | -4.500 | 2.019x10^-4 | 1.026x10^-5 | -8.234x10^-7 | 2.758x10^-9 | 0.726    |
| Higgs   | -2.0 | 1.25 | 2.375  | -3.717x10^-4 | 1.152x10^-5 | 4.349x10^-7  | -1.148x10^-9 | -0.302   |
| Z       | -0.5 | -0.75| -4.500 | -1.033x10^-4 | -7.474x10^-6 | -8.217x10^-7 | 4.408x10^-9  | 1.160    |
| W       | 1.5  | -1.75| -2.625 | 2.714x10^-4  | -1.600x10^-5 | -4.758x10^-7 | 6.150x10^-9  | 1.618    |

The residuals are O(1) in Delta^6 units. The W channel carries the largest positive sixth-grade residual and is the unique full-flag endpoint. The sixth grade is a representation boundary term. The Representation Boundary section gives the lifted closure details.

### 4.4 Coefficient Admissibility

No coefficient is continuously fitted to mass data. Each family is placed in this category set:

| Coefficient family | Status in this report | Why |
| ------------------ | --------------------- | --- |
| a_i                | forced                | fixed from `|H|, C1, C2, C3, M_shell` |
| b_i                | forced                | fixed from gauge or matter-offset rules |
| c_i                | selected from finite set | fixed values from stage-projector spectrum `{+Q,-M_shell/8,-M_shell/8+C1/4,-M_shell/8+C1/4-C3/2}` |
| p_i                | forced                | fixed by K4 edge increments and trace-free constraint sum(p_i)=0 |
| q_i                | forced                | fixed by K4 edge increments and trace-free constraint sum(q_i)=0 |
| r5_i               | selected from finite set | fixed by the code formula `r5 = -(C2-C1)/2 + (|H|-(C2-C1))/8*(base-rot) + C2/8*bal` over base/rot/bal flags |
| sqrt(5) factor      | imported fixed normalisation | fixed by K4/32-bit normalisation convention |

### 4.5 Structural Independence Audit

The numerical closure is meaningful only if the tested algebraic object is fixed before comparison with the electroweak masses. The following audit states exactly what is fixed internally, what is supplied as an interpretive channel assignment, and what remains open.

| Item | Required clarification | Current status in this report |
| ---- | --------------------- | ---------------------------- |
| Delta | Was it fixed before top/H/Z/W evaluation? | `Delta = 1 - d_BU/m_a` is fixed from the BU monodromy and aperture constants in the finite-kernel layer and is independent of the electroweak mass set used in Section 5. |
| Channel assignment | Are Top/H/Z/W flags independently forced or post hoc selected? | Within the fixed flag grammar and after imposing trace-free `sum p_i = 0, sum q_i = 0`, the declared Top/Higgs/Z/W assignment is the unique rank-1 candidate among 96 admissible candidates under the max-absolute-tick-error metric. |
| Coefficient grammar | Which terms are unique versus finite selections? | `a_i, b_i, p_i, q_i` are forced in this setup. `c_i` and `r5_i` are selected from a finite grammar domain once the channel flags are fixed. `sqrt(5)` is an imported fixed normalisation factor from the K4/32-bit convention. |
| Null model | How many nearby grammars could produce comparable residuals? | The discrete family has `2^12 = 4096` flag assignments; trace conditions `sum p_i = 0, sum q_i = 0` reduce this to 96 admissible candidates. Section 5.0 ranks these by max abs tick error before any external-channel interpretation. |

This section therefore tests a fixed, predeclared grammar object and then assesses empirical compression strength under that object.

 

## 5. Electroweak Numerical Tests

The five-order law derived in Section 4 is the basis for all tests in this section.

### 5.0 Null-Model Audit for the Electroweak Law

The null-model baseline for the electroweak law keeps the grammar fixed and only varies the discrete kernel flags. Each of the four channels uses:

```text
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```

with:

* same shell-based a_i and b_i formulas,
* same stage projector family for c_i,
* p_i and q_i generated from base/rot/bal flags through K4 edge increments,
* r5_i generated from the same base/rot/bal flags through the r5 formula.

This gives 2^12 = 4096 raw flag assignments. Enforcing trace conditions
sum(p_i)=0 and sum(q_i)=0 yields 96 grammar-consistent candidates.

For each tracefree candidate, the script evaluates the full law at `Delta = DELTA` and ranks it by maximum absolute tick error to the observed electroweak coordinates.

The declared channel assignment `(Top, Higgs, Z, W)` enters as rank:

```text
declared assignment rank = 1
rank 1 max_abs_tick_error = 6.150e-09
rank 2 max_abs_tick_error = 6.955e-05
unique rank factor gain ~ 1.1e4
```

The null-model audit uses the following declared filter table:

| Filter | Surviving candidates |
| ------ | -------------------: |
| Raw flag assignments | 4096 |
| Trace-free p and q conditions | 96 |
| max abs tick error <= 5e-3 | 96 |
| max abs tick error <= 1e-3 | 93 |
| max abs tick error <= 5e-4 | 31 |
| max abs tick error <= 1e-4 | 2 |
| max abs tick error <= Delta^5 closure band | 0 |

The top ranked trace-free candidates are:

| rank | max_abs_tick_error | Top flags | Higgs flags | Z flags | W flags | p_sum | q_sum | sum_abs_err |
| ---: | -----------------: | :-------- | :---------- | :------ | :------ | ----: | ----: | ----------: |
| 1 | 6.150e-09 | (0,0,0) | (1,0,0) | (1,1,0) | (1,1,1) | 0.000 | 0.000 | 1.446e-08 |
| 2 | 6.955e-05 | (1,1,1) | (1,0,0) | (1,1,0) | (0,0,0) | 0.000 | 0.000 | 1.391e-04 |

This is the finite-law-only baseline that keeps the law-core separate from external leads. Ranks 3 through 12 of the declared-filter candidate list are in Appendix A.6.

### 5.1 Mass Prediction and Delta Consistency

Electroweak mass coordinates on the Delta ruler (`E0 = v = 246.22 GeV`):

| Target        | n (ruler coordinate) | log2(EW/m) | Nearest integer | Signed tick residual |
| ------------- | ------------: | ---------: | --------------: | -------------------: |
| Top quark     |     24.695157 |   0.511179 |              25 |  -0.304843 |
| Higgs         |     47.192619 |   0.976866 |              47 |  +0.192619 |
| Z boson       |    69.230400 |   1.433038 |              69 |  +0.230400 |
| W boson       |    78.023779 |   1.615057 |              78 |  +0.023779 |
| Bottom quark  |    284.078644 |    5.880301 |             284 |  +0.078644 |
| Charm quark   |    367.108184 |    7.598976 |             367 |  +0.108184 |
| Tau           |    343.701809 |    7.114474 |             344 |  -0.298191 |
| Muon          |    540.413826 |   11.186325 |             540 |  +0.413826 |
| Strange quark |    547.824985 |   11.339733 |             548 |  -0.175015 |
| Electron      |    912.009808 |   18.878196 |             912 |  +0.009808 |

At Delta^2 level (code enumerator + stage projections only):

| State   | Predicted n | Observed n | Tick error | Mass error   |
| ------- | ----------: | ---------: | ---------: | -----------: |
| Top     |   24.694955 |  24.695157 |  +0.000202 |  2.897x10^-6 |
| Higgs   |   47.192991 |  47.192619 |  -0.000372 | -5.333x10^-6 |
| Z       |   69.230503 |  69.230400 |  -0.000103 | -1.482x10^-6 |
| W       |   78.023508 |  78.023779 |  +0.000271 |  3.894x10^-6 |

At Delta^5 level, the maximum tick error is 6.15x10^-9.

Delta backsolve (four channels plus W/Z):

| Source | Equation | Delta_back | Delta error |
| ------ | -------- | ---------: | ----------: |
| Top | L_t = 73Delta - 1 + Delta^2/4 | 0.020699611150 | 5.724x10^-8 |
| Higgs | L_H = 96Delta - 1 - 24Delta^2 | 0.020699472926 | -8.099x10^-8 |
| Z | L_Z = 117Delta - 47/48 - 45Delta^2/2 | 0.020699535494 | -1.842x10^-8 |
| W | L_W = 126Delta - 47/48 - 65Delta^2/2 | 0.020699598986 | 4.507x10^-8 |
| W/Z | log2(m_Z/m_W) = Delta(9 - 10Delta + 2Delta^2/sqrt(5) - Delta^3) | 0.020699554747 | 8.340x10^-10 |

Four-point mean Delta_back = 0.020699554639 (error 7.255x10^-10 vs reference Delta = 0.020699553913).

The Q=1/4 matter-density term brings the top Delta backsolve error into the H/Z/W cluster (ratio drops from 18.8x to 0.7x the H/Z/W maximum).

### 5.2 Leave-One-Out Test

Each H/Z/W mass is predicted from the other two:

| Target | Delta source |  m_pred (GeV) |  m_ref (GeV) |    rel_err   |
| ------ | ------------ | -------------: | ------------: | -----------: |
| Higgs  | Z + W        | 125.099223017  | 125.100000000 | -6.211x10^-6 |
| Z      | Higgs + W    |  91.187596612  |  91.187600000 | -3.715x10^-8 |
| W      | Higgs + Z    |  80.379658228  |  80.379000000 |  8.189x10^-6 |

### 5.3 W/Z Ratio Lock

The Delta^2 backbone is `log2(m_Z/m_W) = Delta[(C2 - C1) - (C3/2)Delta]`. The promoted D4 law is:

```text
log2(m_Z/m_W) = Delta[(C2 - C1) - (C3/2)Delta + 2Delta^2/sqrt(5) - Delta^3].
```

| Quantity | Predicted | Observed | Error |
| -------- | ---------: | -------: | ----: |
| n_W - n_Z | 8.793378828287 | 8.793379174256 | 3.460x10^-7 |
| sin^2 theta_W | 0.223013217613 | 0.223013225327 | -7.714x10^-9 |
| W from Z and Delta | 80.379000399 GeV | 80.379000000 GeV | 4.964x10^-9 rel |

The W/Z-to-Delta lock gives Delta_back = 0.020699554747 with absolute error 8.34x10^-10 against CGM Delta.

### 5.4 Coupling Parametrization

Couplings are algebraic consequences of the mass law at tree level (`y_t = 2^(3/2 - 73Delta - Delta^2/4)`, `lambda_H = 2^(1 - 192Delta + 48Delta^2)`, `g_Z = 2^(95/48 - 117Delta + 45Delta^2/2)`, `g = 2^(95/48 - 126Delta + 65Delta^2/2)`).

| Quantity | Law | Compact value | Reference value | Relative error |
| -------- | --- | ------------: | --------------: | -------------: |
| lambda_H | m_H^2/(2v^2) | 0.129072386 | 0.129073762 | -1.067x10^-5 |
| g | 2m_W/v | 0.652906450 | 0.652903907 | 3.894x10^-6 |
| g_Z | 2m_Z/v | 0.740699089 | 0.740700187 | -1.482x10^-6 |
| g' | sqrt(g_Z^2-g^2) | 0.349783231 | 0.349790301 | -2.021x10^-5 |
| e | gg'/g_Z | 0.308324569 | 0.308329144 | -1.484x10^-5 |
| alpha_EWDelta | 4pi/e^2 | 132.188476676 | 132.184554083 | 2.968x10^-5 |
| y_t | sqrt(2)m_t/v | 0.992284310 | 0.992281435 | 2.897x10^-6 |

## 6. Representation Boundary and the 32-bit Lift

### 6.1 The 32-bit Necessity

The 24-bit carrier space Omega fails the first-order spectral triple condition, the SU(3) sextet bracket, and the sixth-grade W-boundary closure. The 24-bit space identifies S-gate pairs {0xAA, 0x54} and {0xD5, 0x2B}, collapsing four family phases into two spatial actions. Depth-4 family fiber probing gives 256 assignments collapsing to 4 distinct 24-bit outputs. In the 32-bit lifted space the spectral triple closes on the K4 gauge subalgebra, the SU(3) sextet bracket closes under family-phase symmetrization, and the W sixth-grade residual closes as a path-multiplicity resonance. The 32-bit lift is the minimal representation required for structural consistency.

### 6.2 Finite Colour Operator Algebra

The 15 weight-4 codewords decompose as 1 + 6 + 8. The adjoint bracket and phase-symmetrized sextet bracket close, and paired adjoint action preserves relational bulk support.

| Check                 | Result |
| --------------------- | ------ |
| 1+8+6 decomposition  | closes |
| Adjoint bracket       | closes |
| Sextet bracket        | closes (phase-symmetrized lifted check) |
| Paired action preserves bulk | yes |
| Left-action leaks     | yes    |

Left-action leakage under single-sided adjoint action is retained as a control confirming that confinement requires the paired conjugate structure.

### 6.3 Executable Closure Probes

Executable closure probes are in Appendix A.

## 7. Lepton Carrier Layer

The lepton coordinates anchor to the horizon and shell structure through the M_shell moment:

```text
n_lepton = k * |H| + r(M_shell) + corrections
```

| Lepton  | k  | r(M_shell) | Base | n_model D3 | n_observed | tau         |
| ------- | -- | ---------: | ---: | ---------: | ---------: | ----------: |
| Tau     | 5  |         24 |  344 | 343.701809 | 343.701809 | 4.931377684 |
| Muon    | 8  |         28 |  540 | 540.413826 | 540.413826 | 7.753769717 |
| Electron| 14 |         16 |  912 | 912.009808 | 912.009808 |13.085368466 |

The base offsets are:

```text
r(tau)      = M_shell/8          = 24
r(muon)     = M_shell/8 + M_shell/48   = 28
r(electron) = M_shell/8 - M_shell/24   = 16
```

The ladder separations are:

```text
tau -> muon      k: 5 -> 8    delta_k = 3 = C1/2
muon -> electron k: 8 -> 14   delta_k = 6 = C1
tau -> electron  k: 5 -> 14   delta_k = 9 = 3*C1/2
```

The muon coordinate uses the equatorial code multiplicity at first order:

```text
n_mu = 540 + C3*Delta = 540 + 20*Delta.
```

An executable horizon-wrap exhaustion probe tests whether these anchors are forced by the current algebraic conditions. With only optical ordering and the tau/muon 64-cost budget, there are 680 valid increasing k triples up to k <= 16. Adding the candidate horizon-wrap rule:

```text
k_tau = q_source = 5
k_mu - k_tau = C1/2 = 3
k_e - k_mu = C1 = 6
```

selects the unique path (5, 8, 14). Broad carrier-budget conditions alone yield 680 valid triples; imposing the horizon-wrap rule selects exactly one. The lepton anchor path is therefore the unique path under the combined conditions.

### 7.1 Carrier Algebra

Let M_q be the shell transition matrix at Hamming weight q, let Tr(M_q) be its trace, and let Tr(M_q^2) be the return trace sum over i,k of (M_q)_{ik}(M_q)_{ki}. Define the carrier trace C(q) to be Tr(M_q) when it is non-zero, otherwise Tr(M_q^2).

The following identities are exact rational consequences of the transition law:

| q | Tr(M_q) | Tr(M_q^2) | C(q)   | byte count 4*C(6,q) |
| - | ------- | --------- | ------ | --------------------: |
| 0 | 7       | 7         | 7      |                     4 |
| 1 | 0       | 28/9      | 28/9   |                    24 |
| 2 | 7/3     | 511/225   | 7/3    |                    60 |
| 3 | 0       | 52/25     | 52/25  |                    80 |
| 4 | 7/5     | 511/225   | 7/5    |                    60 |
| 5 | 0       | 28/9      | 28/9   |                    24 |
| 6 | 1       | 7         | 1      |                     4 |

The q=2 and q=4 shells have the same return trace:

```text
Tr(M_2^2) = Tr(M_4^2) = 511/225.
```

Their linear carrier traces differ:

```text
C(2) = 7/3
C(4) = 7/5
```

For the lepton Delta^3 carrier differences:

```text
C(2) - C(4) = 14/15        (muon and electron)
C(4) - C(5) = -77/45       (tau)
```

The byte counts (24, 60, 60) at q in {5,4,2} show that the temporal ladder moves from the 24-byte q=5 support stratum into the 60-byte q=4 and q=2 strata.

Carrier traces give the Delta^3 corrections. The temporal ladder uses three rational carrier weights:

```text
tau:      coeff = -27/64   on (q=5 -> q=4)
muon:     coeff = -37/64   on (q=4 -> q=2)
electron: coeff = -51/256  on (q=4 -> q=2)
```

The correction has the form:

```text
resid_D3 = coeff * (C(q_to) - C(q_from)) * Delta^3.
```

The exact Delta^3 coefficients are:

```text
tau:      (-27/64)*(C4-C5) = 231/320
muon:     (-37/64)*(C2-C4) = -259/480
electron: (-51/256)*(C2-C4) = -119/640
```

The dyadic numerators decompose into compact kernel quantities:

| Lepton   |  64-normalised cost | Rule               |
| -------- | -----------------: | ------------------ |
| Tau      | -27                | -(C1/2)*(C2-C1)  |
| Muon     | -37                | `-(|H|-27)`        |
| Electron | -51/4              | `-(3*|K4|+C1/8)`   |

The tau and muon costs saturate the horizon budget:

```text
|tau| + |muon| = 27 + 37 = 64 = |H|
|muon| - |tau| = 10 = C3/2
```

Static traces alone cannot split muon from electron: both use C(2)-C(4)=14/15 but the implemented dyadic ratio is:

```text
(-37/64) / (-51/256) = 148/51.
```

The squared-carrier ratio is:

```text
C(2)^2 / C(4)^2 = (7/3)^2 / (7/5)^2 = 25/9.
```

These rational numbers are not equal. Static carrier weights alone do not generate the muon/electron split. The q=2 and q=4 supports are isospectral at the return-trace and Hilbert-Schmidt levels; the Hilbert-Schmidt probe table is in Appendix A.7. The muon/electron separation requires temporal path information in addition to static q-support data.

The 32-bit lift summary provides an explicit transition-history closure for this split:

```text
K4 depth-4:      max family paths/output = 128
full-byte len-2: max family paths/output = 16
full-byte len-2: max micro paths/output  = 4
equatorial code constants: C3 + C2       = 20 + 15

numerator   = 128 + 16 + 4   = 148
denominator = 20 + 15 + 16   = 51
closure     = 148/51         (exact)
```

The q-history path plus byte reset resolves the split. The Delta^3 carrier edges are:

```text
tau:             q=5 -> q=4
muon/electron:   q=4 -> q=2
```

The edge union has a unique directed chain:

```text
q=5 -> q=4 -> q=2.
```

The q-history moment, affine path fit, and intermediate split algebra are in Appendix A.7. The implemented electron coefficient includes the byte-horizon reset:

```text
byte reset = -(C1/2)/256 = -3/256.
```

Thus:

```text
-3/16 - 3/256 = -51/256.
```

Equivalently, the electron cost is:

```text
-51/256 = -(3*|K4| + C1/8)/64.
```

The reset uses the three interior gyro roles over the 256-byte horizon.

### 7.2 Archetype Closure

The electron dyadic is carrier-neutral up to the archetype byte shadow. The exact carrier-neutral completion would choose the electron dyadic that makes the Delta^3 carrier coefficient sum vanish:

```text
tau_coeff + muon_coeff + electron_coeff = 0.
```

Solving this condition gives:

```text
electron_neutral = -25/128 = -50/256.
```

The implemented electron coefficient is:

```text
electron_implemented = -51/256.
```

The difference is one byte tick:

```text
electron_implemented - electron_neutral = -1/256.
```

The corresponding residual carrier sum is:

```text
tau_coeff + muon_coeff + electron_coeff = -7/1920.
```

The residual is exactly the shared q=4 to q=2 carrier delta multiplied by the single archetype byte atom:

```text
C(2) - C(4) = 14/15
GENE_MIC_S = 0xAA
0xAA xor 0xAA = 0x00
archetype atom = 1/256
-(1/256)*(14/15) = -7/1920.
```

The byte 0xAA is the unique zero-intron S-gate in q^2(0): the archetype atom that closes the carrier graph at electron_neutral - 1/256 = -51/256. The horizon gate selection audit is in Appendix A.

### 7.3 Electron Residual and Top Density

The electron residual beyond n_e = 912 decomposes as:

| Term                       |          Value |      Share |
| -------------------------- | -------------: | ---------: |
| SU(2) residual sigma           | 0.009396010431 | 95.80% |
| Higgs-memory term (5/256)/n_H | 0.000413862387 |  4.22% |
| Sum                        | 0.009809872818 |      100.02% |
| Observed residual          | 0.009808220695 |       N/A  |
| Match error                | 1.652e-06 |       N/A  |

Both terms are derived from compact definitions in the framework.

The top Q=1/4 test: linear-only L_t gives Delta error 1.525x10^-6 (18.8x H/Z/W max); with Delta^2/4 the error is 5.724x10^-8 (0.71x). The Q=1/4 matter-density term is structurally required.

## 8. Colour/Strong-Sector Diagnostics

### 8.1 Residual Basis

Symbols epsilon, eta, and d_H are defined in Section 3. This section retains omega, kappa, and sigma:

| Symbol | Definition | Value |
| ------ | ---------- | ----: |
| omega | d_BU/2 | 0.097671088290 |
| kappa | pi/4 - 1/sqrt(2) | 0.078291382211 |
| sigma | (phi_SU2 - 3d_BU)/m_a | 0.009396010431 |

### 8.2 Residual Closures

| Target   | Compact selector      | Predicted n | Observed n | Tick error | Mass error   |
| -------- | --------------------- | ----------: | ---------: | ---------: | -----------: |
| Top      | 25 - epsilon + Delta/4      |   24.694955 |  24.695157 |  +0.000202 |  2.897x10^-6 |
| Higgs    | 48 - d_H              |   47.192991 |  47.192619 |  -0.000372 | -5.333x10^-6 |
| Z        | 70 + Delta - P*d_H        |   69.230503 |  69.230400 |  -0.000103 | -1.482x10^-6 |
| W        | 79 - 9Delta - P*d_H       |   78.023508 |  78.023779 |  +0.000271 |  3.894x10^-6 |
| Bottom   | 284 + kappa               |  284.078291 | 284.078644 |  +0.000352 |  5.055x10^-6 |
| Charm    | 367 + omega + Delta/2     |  367.108021 | 367.108184 |  +0.000163 |  2.341x10^-6 |
| Strange  | 548 - (omega + kappa)         |  547.824038 | 547.824985 |  +0.000948 |  1.360x10^-5 |
| Tau      | 344 - (d_BU + 5Delta)     |  343.701160 | 343.701809 |  +0.000649 |  9.313x10^-6 |
| Muon     | 540 + 20Delta             |  540.413991 | 540.413826 |  -0.000165 | -2.374x10^-6 |
| Electron | 912 + sigma + (5/256)/n_H |  912.009810 | 912.009808 |  -0.000002 | -2.361x10^-8 |

Quark masses are scale-dependent and scheme-dependent. The table entries should therefore be read as compact-coordinate selectors appropriate to the stated convention, and their numerical values depend on the chosen scheme.

### 8.3 Quark Boolean Lattice

Bottom, Charm, and Strange occupy the three empirical selector positions on the {kappa, omega} lattice, while Top is treated separately as the affine UV-anchored channel:

| Quark   | kappa present | omega present | Selector          |
| ------- | --------- | --------- | ----------------- |
| Top     | no        | no        | 25 - epsilon + Delta/4  |
| Bottom  | yes       | no        | 284 + kappa           |
| Charm   | no        | yes       | 367 + omega + Delta/2 |
| Strange | yes       | yes       | 548 - (omega + kappa)         |

Only these three hadron-sector selectors are represented on the empirical lattice in this report.

### 8.4 kappa/omega Consistency Test

The closed-form values kappa = π/4 - 1/sqrt(2) and omega = d_BU/2 are tested against direct estimators from quark coordinates:

| Estimator              | Value          | Closed form    | Residual      |
| ---------------------- | -------------: | -------------: | ------------: |
| kappa from Bottom          | +0.078643732   | +0.078291382   | +3.524x10^-4 |
| omega from Charm           | +0.097834230   | +0.097671088   | +1.631x10^-4 |
| kappa+omega from Strange       | +0.175014606   | +0.175962470   | -9.479x10^-4 |

Internal consistency (no closed form): (kappa+omega)_s - (kappa_b + omega_c) = -1.463x10^-3.

PDG uncertainty floor is approximately 1% on bottom/charm and 5% on strange, mapping to Delta-units of order 0.7. The 10^-3 residuals are at or below experimental noise, whereas the H/Z/W cluster sits at approximately 10^-7 because their masses are known far more precisely.

### 8.5 D_flow^2 Quark Ladder

An empirical chirality-flow probe gives a strict square pattern for six quark entries:

| Quark | mass (GeV) | log2(mass) | d_flow^2 | |d_flow| |
| ----: | ----------: | ----------: | -------: | ------: |
| Up        | 0.002160 | -8.854753 | 1   | 1 |
| Down      | 0.004670 | -7.742362 | 4   | 2 |
| Strange   | 0.095000 | -3.395929 | 9   | 3 |
| Charm     | 1.270000 |  0.344828 | 16  | 4 |
| Bottom    | 4.180000 |  2.063503 | 25  | 5 |
| Top       | 172.760000 | 7.432625 | 36  | 6 |

The pattern is exact squared spacing in d_flow magnitude: `|d_flow| = 1..6`.

### 8.6 C3 Equatorial Attenuation Proxy

The same executable output yields the equatorial attenuation ladder:

```text
attenuation ratios            = 3/4, 1/2, 1/4
attenuation tick scales       = 20.050553, 48.310220, 96.620440
alpha_s proxy at C3           = 0.881564588857
equivalent n_f from b0        = 13.441506
```

These values are a finite proxy for compact strong-sector diagnostics.

Strong-scale ruler placement. With Lambda_QCD = 0.2000 GeV (conventional input), `n_QCD = log2(v/Lambda_QCD)/Delta = 495.939781`. The electroweak and top-anchor constructions agree at displayed precision. The strong scale lies deep in the relational bulk (3968 states between the 128 horizon states).

### 8.7 UV-IR Conjugacy

The CGM stage energies satisfy `E_UV x E_IR = K` with `K = E_CS x E_EW / (4pi^2)`. The stage products close numerically:

| Stage | UV energy (GeV) | IR energy (GeV) | UV x IR / K |
| ----- | --------------: | --------------: | ----------: |
| CS | 1.221e19 | 6.237 | 1.000000000 |
| UNA | 5.496e18 | 13.855 | 1.000000000 |
| ONA | 6.104e18 | 12.474 | 1.000000000 |
| BU | 3.093e17 | 246.220 | 1.000000000 |

Carrier-trace ratios from UV-IR shell coupling match explicit conjugacy: Top/electron (q=2,4) ratio 0.600000; Higgs/muon (q=3,4) ratio 0.673077.

## 9. Research Leads

### 9.1 Null-model protocol

All channels in this section are treated as hypothesis-generating leads until they pass a fixed null-model audit under a predeclared search grammar.

Required audit items:
1. predeclare the tested functional family, coefficient domain, and tolerance window before evaluating data;
2. report total candidate count, accepted-hit count, and multiplicity correction for multi-form searches;
3. evaluate against a baseline null process that preserves the relevant marginals (catalog density, measurement errors, and sampling windows);
4. publish permutation or bootstrap p-values and false-discovery-rate adjusted q-values;
5. classify channels as evidential only when corrected significance and out-of-sample replication are both satisfied.

Executable CKM and QCD null audits reported in this work are:

```text
CKM metric observed     = 1.968e-04
CKM null mean          = 8.777e-02
CKM p-value            = 0.035
CKM q-value            = 0.071
CKM status             = screen-passed

QCD metric observed     = 4.111
QCD null mean          = 8.360
QCD p-value            = 0.249
QCD q-value            = 0.249
QCD status             = screen-not-passed
```

### 9.2 CKM Compact Ansatz (lead)

The CKM ansatz uses compact angular modes:

| Quantity  | Law               |     Predicted |      Reference |         Error |
| --------- | ----------------- | ------------: | -------------: | ------------: |
| |V_us|  | sin(d_BU + 3Delta/2)  | 0.224462579   |    0.224300000 |  1.626x10^-4 |
| |V_cb|  | sin(2Delta)           | 0.041387283   |    0.040800000 |  5.873x10^-4 |
| |V_ub|  | sin(9Delta^2)          | 0.003856234   |    0.003820000 |  3.623x10^-5 |
| |V_ub incl.| sin(9Delta^2 + phase_shift) | 0.004128961446 | 0.004130       | -1.039e-06 |

Here phase_shift = 0.000272729388 from the inclusive/exclusive offset correction used in the run output. The inclusive residual is approximately a phase-shifted version of the same 9Delta^2 mode.

The element |V_ub| is governed by the Delta^2 mode 9*Delta^2, matching the same second-order porosity structure that appears in the electroweak sector. The CP phase ansatz is delta_CKM = p/2 - 18Delta = 68.652 deg. A full Wolfenstein-coordinate summary (rho, eta_CKM, J) is deferred.

### 9.3 Atomic Spectroscopy Alignment (lead)

Same-element spectral line pairs align to compact levels. The compact levels map onto the kernel's code chart:

| Level | Compact role                              | Best pair      | Error (ticks) |
| ----: | ----------------------------------------- | -------------- | ------------: |
|    12 | constitutional diameter                   | He 10917/12968 |         0.001 |
|    16 | mask-code weight 2                        | Cs 8047/10124  |         0.001 |
|    32 | mask-code weight 4                        | Na 2839/4494   |         0.000 |
|    48 | mask-code weight 6 / depth-4 frame        | Na 3094/6161   |         0.008 |
|    64 | mask-code weight 8 / horizon size         | Cs 5466/13693  |         0.006 |
|    80 | mask-code weight 10                       | Na 2905/9154   |         0.001 |
|    96 | mask-code weight 12                       | He 4713/18685  |         0.001 |

Closed-core/shell systems (He, Na, Cs) align more sharply than hydrogen. This contrast identifies a structural lead for further derivation.


### 9.4 Antihydrogen Aperture Tests (lead)

The mirror-tick coordinate for hydrogen/antihydrogen hyperfine comparison is eta_X = log2(nu_H/nu_Hbar)/Delta. Current 10^-4-scale sensitivity is within a factor of approximately 1.35 of the sigma tick scale (9.4x10^-3 ticks). A moderate improvement would probe the SU(2)-residual mirror scale directly.

The compact gravity aperture uses the constitutional diameter 12: 12*Delta = 0.2484. The residual from quarter closure is 1/4 - 12*Delta = 0.001605. The predicted a_Hbar/g = 1 - 12*Delta = 0.7516, compared to the dyadic 3/4 = 0.7500. Current measurement precision (0.206 combined uncertainty) is far from resolving Delta-level deviations.


### 9.5 Redshift on the Delta Ruler (lead)

The redshift coordinate is placed on the same attenuation ruler used in the compact electroweak sector:

```text
n(z) = ln(1+z) / (Delta ln2),    1+z = 2^(n Delta),    phase(z) = n(z) mod 48.
```

This map is exactly invertible and therefore gives a consistent shared coordinate language across micro, meso, and macroscopic scales. The z=1 landmark maps to n(1) = 1/Delta = 48 + epsilon, where epsilon = 1/Delta - 48 is the same conversion gap already present in the compact coordinate system.

In this report:
1. coordinate placement is established;
2. phase-residue testing is proposed as a falsifiable lead;
3. cosmological dynamical derivation remains outside the present closure scope.

The concrete test channel is residual structure versus phase(z) rather than versus only smooth functions of z. If the attenuation interpretation is physical, precision observables should show nontrivial phase dependence at aperture-frame boundaries.


### 9.6 Strong-Scale Ruler and Bulk Confinement Lead

Ruler placement for Lambda_QCD is in Section 8. The bulk-confinement lead: the strong scale lies deep in the relational bulk of Omega. The two horizons contain 128 states in total, while the bulk contains:

```text
|Omega_bulk| = 4096 - 128 = 3968.
```

In this reading, confinement should be sought in transformations that preserve relational bulk support instead of transitions that terminate on the equality or complement horizons. A possible color-algebra route is to map SU(3) colour to the threefold organisation already present in the compact grammar:

```text
16^3 = 4096
APERTURE_FRAME = 3*|K4|^2 = 48
GF(2)^6 = three paired binary axes.
```

### 9.7 Sector Pattern

| Region | Representative observables | Compact behaviour |
| ------ | -------------------------- | ------------------- |
| UV/backbone | top, Higgs | close to bare or Delta-linear structure |
| Transition | W/Z | Delta plus Delta^2 porosity correction |
| IR/lepton | electron, muon, tau | dressed Delta and SU(2)-residual sensitivity |

**Code-valued curvature (lead).** The commutator defect lands in the self-dual [12,6,2] code. Loop residue is code-valued. Testable development: formulate q as a discrete connection and classify minimal loops.

**Holographic thermal floor (lead).** The identity |H|^2 = |Omega| implies minimum effective support |H| = 64. A compact holographic system cannot reduce effective support below boundary cardinality.

## 10. Conclusions

1. The compact electroweak core reduces the top, Higgs, Z, and W mass coordinates to a carrier-trace polynomial on the Delta aperture ruler.

2. The exact kernel supplies the coefficient grammar: `|Omega|` = 4096, dual `|H|` = 64 horizons, seven-shell binomial classes, the self-dual [12,6,2] code chart, C1=6, C2=15, C3=20, M_shell=192, and the projectors P=47/48, Q=1/4.

3. The electroweak coordinate law closes the four channels with fixed discrete coefficients. The strongest ratio-channel result is the corrected W/Z split, recovering Delta to 8.34x10^-10 and predicting W from Z and Delta at 5x10^-9 relative error.

4. The Delta^6 residuals are order-unity boundary markers. The W channel carries the largest positive residual. The sixth-grade sector is a representation boundary requiring the 32-bit lift.

5. External channels (Section 9) are retained as hypothesis channels pending predeclared null-model audits.

Scope boundaries and deferred channels are listed in Section 9.

## Reproducibility manifest

All numeric tables and probe summaries are generated from the executable workflow in this repository:

* `experiments/aqpu_compact_geom_core.py`
* `experiments/aqpu_compact_geom_kernel.py`
* `experiments/aqpu_compact_geom_report.py`

The script-level inputs are fixed in this report and echoed in the executable output. Appendix A reproduces the raw probe outputs and extended audit tables used for the closures, null-model ranking, and lepton carrier derivations. No table entries are tuned by hand after generation; formatting only is adjusted for readability.

## Appendix A. Tables and executable probes

### A.1 Source-traceability probe

```text
selected byte = 0xAA (unique in q^2(0), zero intron)
dyadic atom = 1/256; carrier delta C(2)-C(4) = 14/15
electron_neutral = -25/128; electron_implemented = -51/256
closes electron dyadic = yes
```

### A.2 148/51 closure probe

```text
closure numerator = 128 + 16 + 4 = 148
closure denominator = 20 + 15 + 16 = 51
closes exactly = yes
```

### A.3 K6 complement-horizon probe

```text
candidate = K6 = P_6; P6 dimension = 64
W unique full K4 endpoint = yes; W sixth-grade residual = 1.618304
natural K4 character lifts close target = no
rich-K6 lifted completion closes target = yes
```

### A.4 Spinorial shadow obstruction

| Check | Result |
| ----- | ------ |
| Unique 24-bit gate actions | 2 |
| Unique family phases | 4 |
| S-pair same shadow | yes |
| C-pair same shadow | yes |
| Requires 32-bit lift | yes |

### A.5 Horizon gate selection audit

| Byte | Gate | Intron | Family | Micro-ref | q-weight | S-gate | Zero source | Selected |
| ---- | ---- | ------ | -----: | --------: | -------: | ------ | ----------- | -------- |
| 0xAA | S | 0x00 | 0 | 0 | 0 | yes | yes | yes |
| 0x54 | S | 0xFE | 2 | 63 | 0 | yes | no | no |
| 0xD5 | C | 0x7F | 1 | 63 | 0 | no | no | no |
| 0x2B | C | 0x81 | 3 | 0 | 0 | no | no | no |

### A.6 Null-model audit: ranks 3 through 12

Trace-free candidates ranked by maximum absolute tick error (declared channel flags fixed at Top=(0,0,0), Higgs=(1,0,0), Z=(1,1,0), W=(1,1,1)):

| rank | max_abs_tick_error | Top flags | Higgs flags | Z flags | W flags | p_sum | q_sum | sum_abs_err |
| ---: | -----------------: | :-------- | :---------- | :------ | :------ | ----: | ----: | ----------: |
| 3 | 1.989e-04 | (1,0,1) | (1,0,0) | (1,1,0) | (0,1,0) | 0.000 | 0.000 | 3.978e-04 |
| 4 | 2.684e-04 | (1,0,1) | (1,1,0) | (1,1,0) | (0,0,0) | 0.000 | 0.000 | 5.369e-04 |
| 5 | 2.684e-04 | (0,0,0) | (1,1,0) | (1,0,0) | (1,1,1) | 0.000 | 0.000 | 5.369e-04 |
| 6 | 2.684e-04 | (0,1,0) | (1,0,0) | (1,0,0) | (1,1,1) | 0.000 | 0.000 | 5.369e-04 |
| 7 | 2.684e-04 | (1,0,1) | (1,1,0) | (1,0,0) | (0,1,0) | 0.000 | 0.000 | 9.346e-04 |
| 8 | 2.684e-04 | (1,1,1) | (1,0,0) | (1,0,0) | (0,1,0) | 0.000 | 0.000 | 5.369e-04 |
| 9 | 2.684e-04 | (1,1,1) | (1,1,0) | (1,0,0) | (0,0,0) | 0.000 | 0.000 | 6.760e-04 |
| 10 | 2.684e-04 | (0,0,0) | (1,1,0) | (1,1,0) | (1,0,1) | 0.000 | 0.000 | 5.369e-04 |
| 11 | 2.684e-04 | (0,1,0) | (1,0,0) | (1,1,0) | (1,0,1) | 0.000 | 0.000 | 5.369e-04 |
| 12 | 2.684e-04 | (0,1,0) | (1,1,0) | (1,0,0) | (1,0,1) | 0.000 | 0.000 | 1.074e-03 |

### A.7 Lepton carrier algebra: Hilbert-Schmidt probe and q-history fit

Hilbert-Schmidt spectral-weight probe at the q-shells relevant to the muon/electron obstruction:

| q | byte count | Tr(M_q^2) | C(q) | shell Frobenius^2 | full operator Frobenius^2 |
| - | ---------: | --------: | ---: | ----------------: | ------------------------: |
| 2 |         60 |   511/225 |  7/3 |          1001/225 |                     64/15 |
| 4 |         60 |   511/225 |  7/5 |          1001/225 |                     64/15 |
| 5 |         24 |      28/9 | 28/9 |             91/18 |                      32/3 |

q-history moment on the directed chain q=5 -> q=4 -> q=2:

```text
mean popcount(q5 xor q4 xor q2) = 25/9.
```

Affine path fit for the k-step in terms of the running parameter r:

```text
k_step = round(-2.000*r + 8.000)
```

This yields `(5, 8, 14)` via steps `(3, 6)` on the q5->q4 and q4->q2 transitions.

Multiplication by the W/Z offset gives the path split:

```text
-(C2-C1) * 25/9 = -9 * 25/9 = -25.
```

Simplified muon/electron split from the path moment:

```text
(-37/64) - (-3/16) = -25/64.
```

## Appendix B. Formula catalogue

Master electroweak law:

```text
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```
- `S_WZ = log2(m_Z/m_W)`.
- `sin^2 theta_W = 1 - (m_W/m_Z)^2 = 1 - 2^(-2S_WZ)` with `S_WZ = Delta[(C2 - C1) - (C3/2)Delta + 2Delta^2/sqrt(5) - Delta^3]` (translated from W/Z ratio split).
- `n_QCD = log2(v/Lambda_QCD)/Delta`
- `r(tau) = M_shell/8`, `r(muon) = M_shell/8 + M_shell/48`, `r(electron) = M_shell/8 - M_shell/24`

See the main text for the derivation context of each formula and the governing assumptions.

## Appendix C. External empirical inputs

- PDG review masses and widths (as listed in Section 1).
- CODATA/NIST coupling and constants input set.
- Conventional QCD reference scale `Lambda_QCD = 0.2000 GeV`.
- Auxiliary data and probe outputs are listed in Appendix A.

## References

1. Particle Data Group, S. Navas et al., *Review of Particle Physics*, Phys. Rev. D 110, 030001 (2024), with online updates at the PDG website.
2. E. Tiesinga, P. J. Mohr, D. B. Newell, and B. N. Taylor, *The 2022 CODATA Recommended Values of the Fundamental Physical Constants*, Web Version 9.0, NIST, 2024.
3. P. J. Mohr, E. Tiesinga, D. B. Newell, and B. N. Taylor, *CODATA Internationally Recommended 2022 Values of the Fundamental Physical Constants*, NIST, published May 8, 2024.
4. B. Korompilias, *Common Governance Model: Mathematical Physics Framework*, Zenodo DOI: 10.5281/zenodo.17521384.
5. Analysis code: experiments/aqpu_compact_geom_core.py, experiments/aqpu_compact_geom_kernel.py, and experiments/aqpu_compact_geom_report.py.