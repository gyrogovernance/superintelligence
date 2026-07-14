# Analysis: hQVM Percolation Note

## Generator-Restricted Percolation on the hQVM Kernel: Companion Note

**Basil Korompilias**

*Technical note accompanying `Analysis_hQVM_Percolation.md`, July 2026*

---

The findings manuscript [`Analysis_hQVM_Percolation.md`](Analysis_hQVM_Percolation.md) reports generator-restricted percolation on the Common Governance Model (CGM) kernel, including exact transport-rank scaling, register-protocol thresholds, coverage hierarchies, and census saturation under partial byte access. This note develops the surrounding material at greater length than the census document carries, covering audience and cross-references, the observable-to-mechanism structure of the kernel, and the connection between its product geometry and the Hilbert-space lift.

## 1. Scope and intended reader

**Subject classes (arXiv-style):** math-ph; cs.LG; cs.IT; math.PR; math.CO; cs.AI

**Keywords:** Common Governance Model, holonomic quantum virtual machine, generator-restricted percolation, GF(2) transport rank, finite transformation semigroups, matroid rank, self-dual coding theory, exact enumeration benchmark, mechanistic interpretability, causal abstraction, representation learning ground truth, statistical mechanics of learning, mathematical physics.

The percolation kernel is an axiomatically specified finite dynamical system whose labels are computed by exhaustive enumeration on 4096 states. Physical vocabulary in the findings manuscript names verified structure on that system and carries the square-root geometry forced by ancestry preservation. Readers with foundations in mathematical physics, finite algebra, or probability who also work on mechanistic interpretability or representation learning are the primary audience. Percolation and coding-theory readers will find a finite model where cluster size closes on GF(2) transport rank under stated fiber conditions, with closed-form register-protocol thresholds across the hQVM(d) family.

The labels physics-informed, physics-guided, and physics-constrained in current machine learning usage denote PDE-constrained or loss-regularized models such as PINNs. The present work uses axiomatic mathematical physics to specify a finite kernel and its verified census, an intersection closer to exact algebraic ground truth and to cross-listed math-ph and cs.LG manuscripts.

**Reading paths.**

| Reader background | Start here | Then |
|-------------------|------------|------|
| Mechanistic interpretability, AI safety | Findings §1.8, Appendix A | Findings §3 (rank theorem), §5 (threshold hierarchy) |
| Percolation, probability | Findings §1.1-1.2, §4.3.5 | Findings §5-6 (census), Appendix B (protocol) |
| Mathematical physics, CGM | Findings §2-3, `hQVM_Features_Report.md` | `Analysis_hQVM_Wavefunction.md`, `Analysis_Gravity.md` |
| Implementation, reproduction | Appendix B, `experiments/hqvm_percolation_analysis*.py` | `gyroscopic/hQVM/`, superintelligence repository tests |

## 2. Document and repository map

### 2.1 Findings manuscripts (science repository)

| Document | Role | Verification |
|----------|------|--------------|
| [`Analysis_hQVM_Percolation.md`](Analysis_hQVM_Percolation.md) | Generator-restricted percolation census; square-root scaling; Appendix A benchmark specification | `hqvm_percolation_analysis_results.txt`, `hqvm_percolation_analysis_5_results.txt` |
| [`Analysis_hQVM_Percolation_Note.md`](Analysis_hQVM_Percolation_Note.md) | This companion note | Cross-references only |
| [`Analysis_hQVM_Wavefunction.md`](Analysis_hQVM_Wavefunction.md) | Spinorial operators; theorems T1-T10; shadow pairs; word confinement | `hqvm_wavefunction_1.py`, `hqvm_wavefunction_2.py` |
| [`Analysis_Gravity.md`](Analysis_Gravity.md) | Kernel gravity invariants; mask code C64; continuous field bridge | `hqvm_gravity_analysis_*.py`, `hqvm_gravity_runner.py` |
| [`Analysis_Gravity_Note.md`](Analysis_Gravity_Note.md) | Shorter gravitational theory note | Manuscript arguments |
| [`Analysis_Gravity_Quadratic_Note.md`](Analysis_Gravity_Quadratic_Note.md) | Quadratic-gravity inflation prerequisites from combinatorial axioms | Cross-reference to gravity analysis |

### 2.2 Theory, specifications, and verification inventory (science repository)

| Document | Role |
|----------|------|
| [`hQVM_Features_Report.md`](../reports/hQVM_Features_Report.md) | Inventory of verified quantum and physics features; CHSH-Tsirelson certificates; script map |
| [`QuBEC_Theory.md`](../specs/hQVM_QuBEC_Theory.md) | QuBEC formalism and thermodynamic structure |
| [`Gyroscopic_ASI_Foundations.md`](../Gyroscopic_ASI_Foundations.md) | Normative hQVM and SDK specification |
| [`CGM_Paper.md`](../CGM_Paper.md) | Consolidated CGM manuscript |
| [`CommonGovernanceModel.md`](../CommonGovernanceModel.md) | Programme overview and bibliography |

### 2.3 Executable kernel (superintelligence repository)

Canonical implementation: [github.com/gyrogovernance/superintelligence](https://github.com/gyrogovernance/superintelligence).

| Location | Role |
|----------|------|
| `guides/GyroSI_Specs.md`, `guides/GyroSI_Holography.md` | Architecture and holographic foundations |
| `tests/test_hQVM_*.py`, `tests/test_physics_*.py`, `tests/test_moments_physics_*.py` | Kernel pytest verification (~464 documented tests per the Features Report) |
| `hQVM_Tests_Report_*.md`, `Physics_Tests_Report.md` | Test report manuscripts |

The science repository holds a vendored copy at [`gyroscopic/hQVM/`](../../gyroscopic/hQVM/). Percolation experiment scripts live only in science.

### 2.4 Percolation experiment scripts (science repository)

| Script | Output | Scope |
|--------|--------|-------|
| `hqvm_percolation_analysis.py` | `hqvm_percolation_analysis_results.txt` | Monte Carlo byte-fraction census, structural observables |
| `hqvm_percolation_analysis_2.py` through `_4.py` | Same results file | Deterministic gates, null models, word-regime shuffle |
| `hqvm_percolation_analysis_5.py` | `hqvm_percolation_analysis_5_results.txt` | hQVM(d) family; exact rank PMF; square-root gates; asymptotic thresholds |

## 3. The observable-to-mechanism relationship

Ancestry preservation forces the reachable set to factorize as Omega = U x V with |U| = |V| = 2^d. Every observable cluster is therefore a square on this product, and under fiber-complete restriction the cluster size equals (2^r)^2 where r is the GF(2) transport rank of the allowed generator set. The findings manuscript establishes this identity across the hQVM(d) family (Findings §3).

The relationship between what is measured and what generates it becomes explicit at two levels. At the generator level, the four-fold family fiber assigns four distinct bytes to one transport value (Findings §2.6). A measurement of the transport register alone identifies the value, not the byte that produced it. At the cluster level, fiber-incomplete restriction breaks the product form, so equal cluster cardinalities can arise from different transport spans (Findings §3.5). Cluster size then constrains the generating set without determining it.

This is the finite, enumerable form of the situation representation-learning research describes when one measured quantity is consistent with several underlying features. The kernel supplies exact ground truth for both the observable and the mechanism, so a probing method that reports transport rank from squared cluster observables can be scored against enumerated labels rather than against an assumed decomposition. Appendix A.1 separates the two candidate routes, counting active generators and reconstructing the span, through size-controlled strata.

The product structure also fixes where the square root sits. In standard bond percolation the cluster nodes are scalars and square roots appear only as emergent statistics such as root-mean-square displacement or exact critical exponents (Findings §1.3). Here the node set is a product from the outset, so cluster size is a square by construction and the transport root is the primitive object. An external observer measuring 4096 reachable states is measuring a 64-dimensional transport root, squared by bipartite causal closure (Findings §3.5).

## 4. Composition depth and regime change

The same 256 byte operators act at two levels of the multiplicative structure. Single-byte L-type steps operate on the full product U x V and connect across all seven shells when transport diversity is sufficient. Canonical four-byte R-type words act within the root H and confine reachability from horizon anchors to the constitutional poles (Findings §4.1; `Analysis_hQVM_Wavefunction.md`, Theorems T1-T10). The generator set is identical in both regimes. The reachability profile differs by composition depth alone.

Word-level confinement is a verified property of Omega, not a statistical tendency. From a horizon anchor the depth-four word closure keeps trajectories on shells 0 and 6, and the shuffle control in Appendix A.2 destroys this confinement while preserving label statistics. The kernel therefore offers a controlled setting where changing composition depth changes which observables are reachable without changing the underlying operators.

## 5. Collective rank and the parity stratum

Transport rank is a matroid invariant of the byte span. No single byte carries it. The even-weight transport subspace reaches 1024 states, and a single odd-weight byte lifts rank to 6 and reaches all 4096 (Findings §3.4). Two generator sets can share local byte statistics while differing in global reachability by this parity obstruction. The distinction between local structure, which individual bytes carry, and global structure, which only the span carries, has a precise meaning here as the GF(2) rank of the transport span.

## 6. Anchor dependence

The shell reflection rule s maps to 6 - s is global and identical everywhere. The observable consequence depends on the starting anchor: horizon anchors yield the shell profile {0, 6}, and a shell-k bulk anchor yields {k, 6-k} (Findings Appendix A.4). A single global rule produces different confinement patterns from different inputs. Appendix A.4 states two internal hypotheses that agree on training anchors and separate on held-out shells, one a single shared reflection rule and one per-anchor memorization. This pattern is a candidate for structural comparison with context-dependent behavior in learned systems, and the appendix task specifies how to test it.

## 7. The Hilbert lift and continuous representations

The kernel's self-dual mask code C64 lifts to a 12-qubit graph state over GF(2) that factorizes exactly into six independent two-qubit Bell pairs, with CHSH correlators at the Tsirelson bound 2*sqrt(2) (`hQVM_Features_Report.md`, Formal Quantum Certification). Byte stepping remains deterministic exact-integer carrier dynamics on GF(2)^24; the Bell certificates are evaluated on the canonical Hilbert lift of the intrinsic stabilizer code, where the symplectic carrier data define states in a complex amplitude space in the standard stabilizer manner.

This lift is the setting in which the product geometry of Section 3 and the tensor structure of quantum superposition coincide. The reachable set Omega = U x V and the graph-state factorization into Bell pairs are two expressions of the same conjugate product built from the mask code. The square-root relation between cluster size and transport rank is the percolation-side reading of the same structure that yields pairwise entanglement on the lift-side reading. For a reader coming from interpretability, this gives one finite system where the algebraic sense of superposition, meaning many-to-one maps from generators to observables, and the physical sense of superposition, meaning amplitude structure on a tensor product, are the same object viewed through two representations.

The bridge to continuous neural representations is open. Byte dynamics are exact over GF(2), while activations in trained models are continuous. A continuous analog of transport rank, whether the effective dimension of a learned subspace, the matroid rank of quantized features, or another invariant, is the natural next construction between this finite kernel and models trained on Appendix A tasks. The finite case fixes what such an analog must reduce to.

## 8. Benchmark layer (summary of Appendix A)

The findings manuscript Appendix A specifies four supervised task families on the percolation system. Their logic is restated here for readers who reach this note first.

**Task 1, rank recovery.** Input is a 256-bit generator membership vector; the label is transport rank or |Reach|. Size-controlled strata separate counting active bytes from reconstructing the q6 span, so the shortcut and mechanistic routes are distinguishable by construction.

**Task 2, dynamics shuffle.** Condition T uses the true word dynamics; condition S permutes transition tables while preserving marginal statistics. Matched pairs test whether an analysis method tracks the label-to-dynamics map or only correlates with labels.

**Task 3, threshold depth.** Five coverage labels on identical inputs turn on at different byte fractions (Findings Table 14). Separate heads test which coverage criterion a probe encodes.

**Task 4, anchor dependence.** As in Section 6, the task separates a single global rule from per-anchor memorization on held-out shells.

Dataset generation for all four tasks requires only the transition table and breadth-first search (Findings Appendix B).

## 9. Related literature (orientation)

| Field | Connection |
|-------|------------|
| Mechanistic interpretability | Ground-truth benchmarks; causal abstraction; sparse autoencoders; circuit discovery |
| Statistical mechanics of learning | Phase transitions in representation; grokking; spin-glass perspectives on loss landscapes |
| Matroid and coding theory | GF(2) rank; self-dual codes; blocking sets in percolation (McDiarmid, 1981) |
| Random semigroup generation | Generator restriction on finite transformation semigroups |
| Mathematical physics | Product-state observables; spinorial double cover; stabilizer quantum information on finite carriers |

## 10. Citation

Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

Repository: [github.com/gyrogovernance/science](https://github.com/gyrogovernance/science). Companion findings manuscript: [`Analysis_hQVM_Percolation.md`](Analysis_hQVM_Percolation.md).
