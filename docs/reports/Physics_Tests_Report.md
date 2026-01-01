# Kernel Physics Report: Verified Structural Properties and CGM-Linked Invariants

**Repository document type:** Technical report  
**Scope:** Router kernel, compiled atlas artifacts, step law verification, and boundary coverage analysis  
**Primary references:**  
- CGM framework archive (Zenodo): Korompilias (2025), DOI: https://doi.org/10.5281/zenodo.17521384  
- CGM paper and findings: `docs/CGM_Paper.md`, `docs/Findings/Analysis_CGM_Units.md`, `docs/Findings/Analysis_Monodromy.md`  
- Kernel specification: `docs/GGG_ASI_AR_Specs.md`

---

## 1. Purpose and scope

This report documents the structural properties of the GGG ASI Router kernel that have been verified through automated testing. The kernel is a deterministic finite-state coordination system designed to provide shared moments and replayable audit for multi-agent governance. Understanding its internal physics is essential for establishing trust in its behaviour as a coordination substrate.

The report does not claim that bytes are physical qubits or that the kernel is physically equivalent to a quantum system. Instead, it establishes what the kernel provably does as a deterministic finite system, and identifies numerical constants and structural relationships that emerge naturally from its design.

Two classes of results are presented. The first concerns kernel correctness and closure properties, including the state representation, transcription mechanism, mask expansion, step law, inverse stepping, and the depth-two and depth-four algebraic identities. The second concerns kernel-derived invariants that correspond to constants defined in the Common Governance Model framework, including the aperture parameter, the monodromy defect, and the fine-structure constant reconstruction.

All reported results correspond to tests in the physics test suite `tests/test_physics_*.py`.

---

## 2. System definition

### 2.1 Kernel state and update rule

The kernel maintains a 24-bit internal state split into two 12-bit components, denoted A and B. Each input byte triggers a deterministic update through three stages.

In the transcription stage, the input byte is combined with the constant 0xAA using exclusive-or to produce an 8-bit intron. This operation is its own inverse, meaning that applying transcription twice recovers the original byte. The constant 0xAA therefore defines a reference point in the byte space.

In the expansion stage, the 8-bit intron is mapped to a 12-bit mask that will be applied to component A. The expansion function is injective, so each of the 256 possible introns produces a distinct mask. Crucially, component B receives no direct mask; only A is mutated before the gyration step.

In the gyration stage, the two components exchange roles through a swap combined with bitwise complement. The new A component is derived from the complement of the old B component, while the new B component is derived from the complement of the mutated A component. This asymmetric update establishes a fundamental chirality in the system, where the two components play structurally different roles.

The specification for these operations is given in `src/router/constants.py` and `docs/GGG_ASI_AR_Specs.md`.

### 2.2 Compiled atlas artifacts

The kernel's finite state space is captured in compiled atlas artifacts that enable fast lookup and exhaustive verification.

The ontology file contains a sorted list of all 65,536 reachable states, stored as 32-bit unsigned integers. The epistemology file contains a lookup table of shape 65,536 by 256, where each entry gives the index of the next state reached by applying a particular byte to a particular current state. The phenomenology file contains the archetype constants and the precomputed mask table.

Tests that require exhaustive verification operate on these artifacts rather than computing transitions dynamically. This ensures that the tests reflect the actual compiled system that would be deployed.

---

## 3. Verified kernel properties

### 3.1 Representation and transcription

The state representation has been verified to be well-formed and invertible. Packing two 12-bit components into a 24-bit state and unpacking them again recovers the original components exactly. All packed states remain within the 24-bit range.

The transcription constant GENE_MIC_S equals 0xAA as specified. The transcription operation is a bijection on the set of bytes, and applying it twice returns the original input. These properties ensure that every byte value is a valid kernel instruction and that the mapping between bytes and introns is reversible.

### 3.2 Mask expansion

The expansion from 8-bit introns to 12-bit masks has been verified to be deterministic and injective. Each of the 256 introns produces a distinct mask, so no information is lost in this stage.

The bottom 12 bits of the 24-bit mask representation are always zero, confirming that only component A receives a direct mutation. This asymmetry is essential for the chirality property described below.

The precomputed mask table matches the expansion function for all 256 byte values, confirming that the compiled artifacts are consistent with the specification.

### 3.3 Chirality and gyration asymmetry

The gyration step has been verified to be asymmetric in a specific sense. The update is not a simple swap of components A and B; the complement operation is essential.

In the update, the new A component depends only on the previous B component through the complement operation, while the new B component depends on the mutated A component through its own complement. This means that the "right" transition (B becoming the new A) preserves structure in a different way than the "left" transition (mutated A becoming the new B).

This asymmetry corresponds to the Common Source axiom in CGM, which states that right transitions preserve the horizon constant while left transitions alter it. The kernel realises this axiom through the specific structure of its gyration formula.

### 3.4 Exact depth laws and invertibility

The algebraic structure of the kernel has been verified through several exact laws that hold for all states and all byte combinations.

Each per-byte transition is invertible. Given the next state and the byte that was applied, the previous state can be recovered exactly using an explicit inverse formula.

The depth-two decoupling law states that after applying two bytes in sequence, the result can be expressed as the original state with the first byte's mask applied to component A and the second byte's mask applied to component B, independently. This means that at depth two, the effects on the two components separate cleanly.

The depth-four alternation identity states that applying bytes x, y, x, y in sequence returns to the original state for any choice of x and y. This identity also equals the sequence y, x, y, x. The depth-four return is a strong closure property that has no analogue in generic bit-manipulation systems.

The commutation characterisation states that depth-two compositions commute if and only if the two bytes are identical. For any pair of distinct bytes, the order of application matters. This was verified exhaustively for all 65,536 ordered byte pairs.

A trajectory closed form holds for arbitrary-length byte sequences. The final state depends only on the exclusive-or of masks at odd positions, the exclusive-or of masks at even positions, and the parity of the sequence length. The detailed ordering within each parity class does not affect the outcome.

### 3.5 Reference operator and separator behaviour

Byte 0xAA plays a special structural role in the kernel because its intron is zero and therefore its mask is zero.

Applying byte 0xAA twice returns to the original state. This involution property means that 0xAA acts as a reference operator that can be used to define inverses and separators.

The separator lemmas describe how inserting 0xAA adjacent to another byte affects the outcome at depth two. If byte x is applied first and then 0xAA is applied, the result after two steps shows the mask effect in component A only. If 0xAA is applied first and then byte x is applied, the result shows the mask effect in component B only. This provides a mechanism for directing mutations to specific components using only the byte action alphabet.

The inverse of any byte action can be expressed as conjugation by the reference operator. Specifically, the inverse of applying byte x is achieved by applying 0xAA, then x, then 0xAA again. This means that reversal requires no special inverse operator; it can be expressed within the same byte alphabet.

### 3.6 Boundary coverage property

The horizon set consists of states where component A equals the bitwise complement of component B. This set contains exactly 256 states, which is the square root of the full ontology size.

The one-step neighbourhood of the horizon set under all 256 byte actions covers the entire ontology of 65,536 states. Every state in the system is reachable from some horizon state in a single step.

This is a strong boundary-to-bulk coverage property. The boundary (horizon) has size 256, and its immediate neighbourhood under the action alphabet spans the entire bulk. The expansion ratio is 255, meaning that each horizon state fans out to 256 successors (including itself in one case), and collectively these cover the full state space.

This property does not by itself constitute a physical holographic principle, but it provides an exact reachability fact that can be used as a coordination and audit primitive. The horizon forms a natural reference subset from which the entire system is accessible.

---

## 4. Kernel structure revealed by the compiled atlas

The tests in `tests/test_physics_2.py` extract additional structure by analysing the compiled atlas artifacts directly.

### 4.1 Global complement symmetry

The complement map sends each 24-bit state to its bitwise complement. This map has been verified to commute with byte actions on the full ontology for a sample of bytes.

The implication is that the kernel dynamics treats each state and its complement symmetrically. Every trajectory has a "mirror" trajectory obtained by complementing all states. This global symmetry can be used to classify states into complementary pairs.

### 4.2 Closed-form dynamics in mask coordinates

By defining mask coordinates relative to the archetype, the kernel dynamics can be expressed in a particularly clean form.

Let u denote the exclusive-or of component A with the archetype A value, and let v denote the exclusive-or of component B with the archetype B value. In these coordinates, the archetype state corresponds to the origin (0, 0).

The update rule becomes: the new u equals the old v, and the new v equals the old u combined with the byte's mask using exclusive-or. This was verified exhaustively for all 16,777,216 state-byte transitions in the atlas.

This closed form reveals that the kernel is effectively an affine linear system on a 16-bit effective phase space. The coordinates u and v each take 256 values (corresponding to the 256 distinct masks), and the update swaps them while applying a translation to one coordinate.

### 4.3 Commutator acts as a global translation

The commutator of two byte actions can be constructed using the reference operator to implement inverses. Specifically, the commutator K(x,y) applies x, then y, then the inverse of x, then the inverse of y.

This commutator has been verified to act as a state-independent translation. The output state equals the input state combined with a fixed displacement that depends only on the exclusive-or of the two masks. The displacement is applied identically to both the A and B components.

This is operationally significant because commutators do not produce arbitrary or state-dependent outcomes. They generate a structured translation subgroup determined entirely by mask differences. In the language of gauge theory and differential geometry, this would correspond to a flat connection where the holonomy around any loop is determined by a simple algebraic rule.

---

## 5. Kernel-native monodromy and defect statistics

### 5.1 Monodromy construction

Monodromy refers to the phenomenon where traversing a closed loop does not return to the original state but leaves a residual memory of the path taken. The kernel exhibits a clean version of this phenomenon.

Consider the word W(x; y, z) consisting of the bytes x, y, x, z applied in sequence. In the mask coordinates, this word closes in the u coordinate (the base) while shifting the v coordinate (the fibre) by the exclusive-or of masks for y and z.

This was verified on sampled states for multiple (y, z) pairs. The base coordinate returns to its starting value, but the fibre coordinate accumulates a defect. This matches the operational definition of monodromy used in the CGM monodromy analysis documents.

### 5.2 Defect distribution and the CS anchor

The defect weight is defined as the population count (number of set bits) in the exclusive-or of two masks. Over all 65,536 ordered (y, z) pairs, the defect weight distribution has been characterised.

The mean defect weight is exactly 6.0 bits, which is half of the 12-bit mask length. The variance is exactly 5.0. These values are intrinsic to the linear code structure of the mask set.

Using a canonical angle mapping where the cosine of the angle equals one minus the weight divided by six, the mean defect angle is exactly pi over two. This was verified to numerical precision and follows from the palindromic symmetry of the weight distribution.

The value pi over two corresponds exactly to the CS threshold in the CGM framework, which defines the quarter-turn angle that establishes orthogonality. The kernel's mask code structure produces this threshold as an exact mean, not an approximation.

---

## 6. Code structure of the mask set

The mask set has been characterised as a structured linear code in 12-bit space.

### 6.1 Weight enumerator closed form

The weight distribution of the 256 masks matches the generating function (1 + z squared) to the fourth power times (1 + z) to the fourth power. The resulting weight counts for weights 0 through 12 are: 1, 4, 10, 20, 31, 40, 44, 40, 31, 20, 10, 4, 1.

This is not a generic distribution. It reflects the specific structure of the expansion function, which creates four paired positions (contributing the (1 + z squared) factors) and four independent positions (contributing the (1 + z) factors).

### 6.2 Linear code rank and dual constraints

The mask set forms a linear code of rank 8 over the binary field GF(2). This means it is an 8-dimensional subspace of the 12-dimensional bit vector space.

The dual code has 16 elements with weight distribution: one element of weight 0, four of weight 2, six of weight 4, four of weight 6, and one of weight 8. The four weight-2 elements correspond to the parity constraints that tie specific pairs of bits together across the anatomical structure.

### 6.3 Primitive minimal masks

Exactly four masks have weight 1, meaning they flip a single bit in the 12-bit representation. These correspond to four distinct byte values and four specific bit positions within the 2 by 3 by 2 anatomical grid.

These four primitives are the minimal nonzero transformations available in the kernel. They correspond to the generators of independent directions in the mask space.

---

## 7. CGM-linked constants reconstructed from kernel-only quantities

The following constants are computed from kernel-intrinsic quantities using relationships defined in the CGM framework. No external fitting or adjustment is applied.

### 7.1 Discrete aperture shadow

The kernel-native discrete aperture is defined as the probability mass of minimal defect events. Specifically, it is the count of weight-0 and weight-1 masks divided by 256.

The kernel value is 5/256, which equals approximately 0.01953.

The CGM aperture is defined as one minus the ratio of the BU monodromy defect to the aperture scale. The CGM value is approximately 0.02070.

The difference is approximately 0.00117, representing a 5.6 percent relative deviation. This gap is consistent with the expected discretisation effects when comparing a 256-element finite code to a continuous rotation group.

### 7.2 Minimal defect angle and hierarchy bridge

The minimal nonzero defect weight is 1 bit. Using the angle mapping, this corresponds to an angle of arccos(5/6), which equals approximately 0.5857 radians.

Dividing by 3 (corresponding to the three-row anatomical structure) gives a kernel delta value of approximately 0.1952 radians. Dividing again by 2 (corresponding to the dual-pole structure) gives a kernel omega value of approximately 0.0976 radians.

The CGM monodromy analysis documents give delta BU as approximately 0.1953 radians and omega as approximately 0.0977 radians.

The differences are approximately 0.00011 radians for delta and 0.00006 radians for omega. These are sub-tenth-of-a-percent deviations achieved without any parameter fitting.

### 7.3 Reconstruction of aperture scale, quantum gravity constant, and fine-structure constant

Using the kernel's closure fraction (one minus the discrete aperture), the aperture scale can be computed as delta divided by the closure fraction. This gives approximately 0.1991.

The quantum gravity constant can be computed as one divided by twice the squared aperture scale. This gives approximately 12.61, compared to the CGM value of 4 pi (approximately 12.57). The difference is about 0.36 percent.

The fine-structure constant can be computed as delta to the fourth power divided by the aperture scale. This gives approximately 0.007296, compared to the CGM paper value of approximately 0.007297. The difference is in the sixth significant figure.

These reconstructions demonstrate that the kernel's discrete combinatorial structure, when interpreted through the CGM framework's normalisation conventions, produces values consistent with the framework's predictions to several significant figures.

---

## 8. Implications for the project

These verified results support several conclusions about the kernel's role in the GGG ASI Router.

First, the kernel is fully deterministic, invertible, and closed on a finite ontology with strong exact algebraic identities. These properties support its use as a coordination substrate where replay and audit are essential requirements.

Second, the kernel's internal mask system is not arbitrary but rather a structured linear code with a closed-form weight enumerator and compact dual constraints. This structure provides a foundation for understanding how information propagates through byte sequences.

Third, the kernel provides a well-defined monodromy mechanism where loops can close in one coordinate while leaving a residual defect in another. This matches the operational definition of monodromy in the CGM framework and provides a geometric primitive for measuring coordination properties.

Fourth, several CGM-linked constants can be reconstructed from kernel-only quantities using the normalisation conventions defined in the CGM documents. The consistency of these reconstructions suggests that the kernel embodies the same geometric structure that the CGM framework describes in continuous terms.

---

## 9. Reproducibility

All claims in this report correspond to tests that can be run locally using pytest with the compiled atlas.

To build the atlas, run `python -m src.router.atlas` which produces the files `ontology.npy`, `epistemology.npy`, and `phenomenology.npz` in the `data/atlas` directory.

To run the physics tests, execute `python -m pytest -v -s tests/test_physics_1.py` for the core kernel properties and `python -m pytest -v -s tests/test_physics_2.py` for the CGM-linked invariant extraction.

The numerical values reported here are taken from an actual test run and should reproduce exactly under the same kernel specification and atlas builder.