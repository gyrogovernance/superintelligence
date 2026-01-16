# Holography Algorithm: Verified Structure of the GGG ASI Router Kernel

**Scope.**  
This report summarizes the holographic structure of the Router kernel as established by three test suites:

- `tests/test_holography.py`
- `tests/test_holography_2.py`
- `tests/test_holography_3.py`

It focuses on:

- The holographic dictionary between boundary (horizon) and bulk (ontology)
- The emergence of the K₄ (tetrahedral) structure on the boundary
- Subregion duality and wedge tiling of the bulk
- Quotient dynamics and the role of the vertex charge χ
- Erasure and reconstruction geometry on the 2×3×2 anatomy
- Provenance degeneracy (history non-uniqueness)
- A cross-domain “meta-ledger” projector
- Boundary stabilizer subcode and coset structure

The propositions below are supported by exhaustive or decisive tests in the listed files.

---

## 1. Architectural Background

The Router kernel operates on a 24-bit state split into two 12-bit components:

- A12: “active” 12-bit component
- B12: “passive” 12-bit component

Not all 2²⁴ states are used. From the archetype, the kernel reaches exactly:

- An ontology Ω of size |Ω| = 65,536 states  

The ontology can be expressed as:

- Ω ≅ C × C

where C is a 256-element linear mask code in 12 bits (`[12,8]` over GF(2)).

A special subset of Ω, the **horizon H**, consists of states fixed by the reference byte `0xAA`. These satisfy:

- A = B XOR 0xFFF

and form a set of size |H| = 256.

The basic holographic scaling follows:

- |Ω| = |H|² = 256²

The tests summarized below characterize, in detail, how H and Ω are related by the kernel dynamics, and how K₄ and χ emerge from the mask code.

---

## 2. Results from `tests/test_holography.py`

### 2.1 Holographic Dictionary Reconstruction

**Test:** `test_holographic_dictionary_reconstruction`  

**Statement.**  
For every state `s = (A,B) ∈ Ω` there exists a unique pair `(h, b)` such that:

- `h ∈ H` is a horizon state with the same A component as s,
- `b` is a byte in `{0..255}`, and
- applying the transition for byte `b` to `h` yields `s`.

Formally:

> ∀ s∈Ω, ∃! (h∈H, b∈Bytes) such that T_b(h) = s.

**Consequence.**  
Every bulk state can be encoded as a boundary anchor in H plus a single byte. This is an exact, invertible holographic dictionary between boundary and bulk.

---

### 2.2 Horizon Walsh Transform and Dual Code

**Test:** `test_horizon_walsh_exact_translation`  

**Statement.**  
Define the set of A-components of horizon states:

- H_A = { A : (A,B) ∈ H }.

The Walsh transform of the indicator of H_A,

- W_H(s) = Σ_{a ∈ H_A} (-1)^{⟨s,a⟩},

takes values:

- W_H(s) = ±256 when s lies in the dual code C⊥, with the sign determined by ⟨s, archetype_A⟩,
- W_H(s) = 0 otherwise.

This matches the structure of the dual code C⊥ of the 12-bit mask code C.

---

### 2.3 K₄ Vertex Decomposition and Subcode Tower

**Test:** `test_horizon_k4_vertex_subcode_tower`  

**Statement.**  

1. The 256 horizon states split into four classes of 64 states each:

   - H = H₀ ∪ H₁ ∪ H₂ ∪ H₃, with |H_v| = 64.

2. Each class corresponds to a K₄ vertex (0,1,2,3).

3. Within the mask code C, there exists a subcode D (size 64, rank 6) such that:

   - Each H_v corresponds to a coset of D in the mask space.

This establishes a subcode tower:

- C (size 256) ⊃ D (size 64)

with the quotient C/D ≅ (Z/2)² labeling the four K₄ vertices.

---

### 2.4 Ecology as Signed Cycle Coherence Across Domains

**Test:** `test_ecology_signed_cycle_coherence`  

**Context.**  
At the ledger level (K₄ edges), each domain ledger `y_D ∈ ℝ⁶` decomposes into gradient and cycle components via Hodge decomposition.  

Let c₁, c₂, c₃ be the cycle components for three domain ledgers.

**Statement.**  
Define:

- G_ij = ⟨c_i, c_j⟩ (Gram matrix on cycle components)
- Ecology index: E = (G₁₂ + G₁₃ + G₂₃)/(G₁₁ + G₂₂ + G₃₃)

The test verifies:

- E is close to +1 for strongly aligned cycles,
- E is close to 0 for independent cycles,
- E is negative for anti-aligned cycles,
- E is invariant under scale changes and under permutations of domains.

This provides a derived, signed, scale-invariant cross-domain coherence measure.

---

### 2.5 Vertex Charge Parity-Check Recovery

**Test:** `test_vertex_charge_has_two_parity_checks_and_vertex_is_affine`  

**Statement.**  

There exist two 12-bit parity-check vectors:

- q₀ = 0x033
- q₁ = 0x0F0

such that for any mask m:

- χ(m) = (⟨q₀,m⟩, ⟨q₁,m⟩) ∈ (Z/2)²

reproduces the K₄ vertex structure on the horizon. The vertex assignment is affine-linear in A.

This gives an explicit equation for vertex labels in terms of the mask code.

---

## 3. Results from `tests/test_holography_2.py`

### 3.1 Vertex Wedges and Subregion Duality

**Test:** `test_vertex_wedges_tile_bulk_subregion_duality`  

**Statement.**  

For each vertex v ∈ {0,1,2,3}, consider:

- the 64 horizon states H_v assigned to vertex v,
- the set W_v = { T_b(h) : h ∈ H_v, b ∈ Bytes }.

Then:

- |W_v| = 16,384 for each v,
- W_v ∩ W_{v'} = ∅ for v≠v',
- ⋃_v W_v = Ω.

**Consequence.**  
Each boundary region (H_v) generates a disjoint bulk wedge (W_v), and the four wedges tile the entire bulk. This is a precise subregion duality at the discrete level.

---

### 3.2 Meta-Hodge Cross-Domain Projector

**Test:** `test_meta_hodge_ecology_18d_cross_domain_cycle_projector`  

**Statement.**  

A K₃ incidence matrix B_meta (for three domains) is used to define a cycle projector P_cycle_edges on its 3-edge space. Lifting via the Kronecker product with I₆ produces:

- P_cycle_18 = P_cycle_edges ⊗ I₆ ∈ ℝ¹⁸×¹⁸

The test verifies:

- P_cycle_18 is idempotent and symmetric,
- rank(P_cycle_18) = 6,
- Applied to [y₀,y₁,y₂] (stacked domain ledgers), it identifies correlated and anti-correlated patterns with exact rational fractions (1/9 and 2/3 for the constructed cases),
- The resulting “cross-domain cycle fraction” is scale-invariant.

This defines a well-behaved cross-domain projector on the 18-dimensional space of three ledgers.

---

### 3.3 Erasure Thresholds and χ-Image Rank

**Test:** `test_erasure_reconstruction_thresholds_2x3x2_geometry`  

**Context.**  
A generator matrix G (12×8) for the mask code C is constructed from intron-basis bytes. For an erasure of bit positions E, the observed positions S are the complement, and:

- rank(G_S) determines the number of consistent codewords: |ambiguity| = 2^(8 - rank(G_S)).

**Erasures considered (size 4 and 6):**

- Row 0: erased bits {0,1,6,7}  
- Frame 0: erased bits {0,1,2,3,4,5}  
- Edges: erased bits {0,5,6,11}  
- Duplication region: erased bits {8,9,10,11}

**Results.**

- Row0, Frame0, Edges each yield:
  - rank(G_S) = 6 ⇒ ambiguity size = 4
- Duplication region {8,9,10,11} yields:
  - rank(G_S) = 8 ⇒ unique reconstruction

For each pattern, the ambiguity subcode E_S (codewords zero on observed bits) is computed, and its χ-image is analyzed:

- χ(E_row0): |χ(E_S)| = 2, χ-rank = 1  
- χ(E_frame0): |χ(E_S)| = 2, χ-rank = 1  
- χ(E_edges): |χ(E_S)| = 4, χ-rank = 2  
- χ(E_dup): |χ(E_S)| = 1, χ-rank = 0

**Interpretation.**

- Some erasures (Row0, Frame0) lose one χ-bit (partial K₄ information).  
- Some (Edges) lose both χ-bits.  
- The duplicated region bits (8–11) are redundant; losing them does not harm reconstruction or χ.

---

### 3.4 Minimum Distance and Decoding Ambiguity

**Test:** `test_minimum_distance_and_decoding_ambiguity`  

**Statement.**  

- The minimum distance d_min(C) of the mask code is 1, because there are four weight-1 codewords.
- An explicit non-codeword is found that has at least two nearest codewords at the same minimal distance.

**Consequence.**  
The mask code is an operation code (contains single-bit primitives) rather than a classical error-correcting code. Nearest-neighbor decoding can be ambiguous; the code is not designed for guaranteed correction radius.

---

### 3.5 Provenance Degeneracy (Non-Cloning of History)

**Test:** `test_non_cloning_provenance_word_history_degeneracy`  

**Construction.**

- Restricted alphabet of 8 generator bytes: `b_i = 0xAA ^ (1<<i)`
- All words of length 6: 8⁶ = 262,144 words
- Applied from the archetype, final states are collected

**Results.**

- Number of distinct final states from these 262,144 words: 4,096
- Each final state has exactly 64 preimage words on average (and by the statistics, this average is exact)
- Conditional entropy H(word | final_state) ≈ 6.46 bits  
- u- and v-coordinate reachable sets each have size 64, giving a 64×64 image in (u,v)-space

**Consequence.**  
Many distinct byte histories lead to the same final state. Final state alone cannot recover the path. Provenance cannot be reconstructed from state; this is quantitatively measured.

---

### 3.6 K₄ Quotient Dynamics

**Test:** `test_k4_quotient_dynamics_theorem`  

**Statement.**  

Let χ be the vertex charge map from Section 2.5. For any state `(u,v)` and mask m_b, define coarse coordinates:

- U = χ(u), V = χ(v), M = χ(m_b)

The test shows that under the kernel transition:

- U' = V
- V' = U XOR M

for all sampled states and all bytes. Furthermore, the set of coarse states (U,V) has size 16.

**Consequence.**  
The full dynamics factors onto a 16-state K₄-based system that follows the same update structure. This is a quotient dynamics or factor system induced by χ.

---

### 3.7 Entanglement Entropy in the Hilbert-Lift

**Test:** `test_entanglement_superposition_hilbert_space_reduced_density`  

**Construction.**  

The mask space C is used as a basis for a bipartite Hilbert space `H_u ⊗ H_v`. For a subset Σ ⊂ C×C, define:

- |ψ_Σ⟩ = (1/√|Σ|) Σ_{(u,v)∈Σ} |u⟩⊗|v⟩

The reduced density matrix on H_u is ρ_u = Tr_v(|ψ_Σ⟩⟨ψ_Σ|), and its von Neumann entropy S(ρ_u) is computed.

Two cases:

1. Separable Σ = U×V with U,V of size 16
2. Graph Σ = {(u, u⊕t) : u ∈ C} for fixed translation t

**Results.**

- For Σ = U×V, S(ρ_u) ≈ 0 bits
- For graph Σ, S(ρ_u) ≈ 8 bits = log₂|C|

**Consequence.**  
The structure obeys standard bipartite entropy behavior: Cartesian-product subsets are unentangled, graph subsets are maximally entangled on the code subspace.

---

## 4. Results from `tests/test_holography_3.py`

### 4.1 Uniform 4-to-1 Frame0 Projection

**Test:** `test_h3_frame0_projection_uniform_4_to_1`  

**Statement.**  

Define the frame0 projection of a mask m ∈ C by:

- frame0(m) = m & 0x3F  (low 6 bits)

Then:

- Every 6-bit pattern in {0..63} appears as frame0(m) for some m in C
- Each such pattern has exactly 4 preimage masks in C
- The four masks for the same frame0 differ only in bits 6 and 7 (a 2-bit index)

**Consequence.**  
Mask-space collapses to 6-bit “micro-references” with a 2-bit family index. This formalizes a 4-family structure over the mask code.

---

### 4.2 Exhaustive Size-4 Erasure Taxonomy

**Test:** `test_h3_exhaustive_erasure_taxonomy_size4`  

**Statement.**  

For all C(12,4) = 495 possible 4-bit erasures E:

- Compute rank(G_S) where S is the complement of E
- Ambiguity size = 2^(8 - rank(G_S))
- Compute χ-rank of the ambiguity subcode E_S

The test prints and verifies the complete histogram over all patterns. Categories include, for example:

- rank(G_S)=8, ambiguity=1, χ-rank=0: 16 patterns  
- rank(G_S)=7, ambiguity=2, χ-rank ∈ {0,1}: 176 patterns  
- rank(G_S)=6, ambiguity=4, χ-rank ∈ {0,1,2}: 246 patterns  
- rank(G_S)=5, ambiguity=8, χ-rank ∈ {1,2}: 56 patterns  
- rank(G_S)=4, ambiguity=16, χ-rank=2: 1 pattern  

**Consequence.**  
This provides a complete classification of 4-bit erasures by their reconstruction rank and quotient-information loss. It is a full “redundancy atlas” for 4-bit erasures of the 12-bit code.

---

### 4.3 Information-Set Threshold

**Test:** `test_h3_min_information_set_size_is_8`  

**Statement.**  

For each subset S ⊆ {0..11} of observed positions, the rank of G_S is computed. Let r_max(s) be the maximum rank over all subsets of size s.

The test finds:

- r_max(s) = s for s=0..7
- r_max(8) = 8
- r_max(s) = 8 for all s ≥ 8

The minimal s with r_max(s) = 8 is s=8.

**Consequence.**  
At least 8 observed bit positions are needed to achieve full rank (and thus unique reconstruction) at the single-step code level.

---

### 4.4 Boundary Stabilizer Subgroup and Vertex Cosets

**Test:** `test_h3_boundary_stabilizer_subgroup_and_vertex_cosets`  

**Statement.**  

Define χ as in Section 2.5. Then:

1. The kernel of χ inside C, D₀ = {m ∈ C : χ(m)=0}, has:
   - |D₀| = 64  
   - rank(D₀) = 6  

2. For each vertex v ∈ {0,1,2,3}, let U_v be the set of u-coordinates of horizon states assigned to v:

   - U_v = { u : (A,B) ∈ H_v, u = A XOR archetype_A }

   Then, for each v,

   - U_v = u₀ XOR D₀  for any u₀ in U_v.

That is, each U_v is exactly one coset of D₀.

**Consequence.**  
Boundary vertex regions are determined by a stabilizer subgroup D₀ of the mask code: one representative plus D₀ reconstructs an entire vertex boundary region.

---

## 5. Summary

Across `test_holography.py`, `test_holography_2.py`, and `test_holography_3.py`, the following properties of the Router kernel are established:

1. **Holographic dictionary:** Every bulk state has a unique decomposition as boundary anchor + byte.

2. **Boundary–bulk tiling:** The 4 K₄ vertex regions on the boundary generate 4 disjoint wedges that tile the bulk ontology.

3. **Algebraic K₄ structure:** K₄ emerges from the mask code as a quotient by a rank-6 subcode, with explicit parity-check vectors and stabilizer subgroup.

4. **Quotient dynamics:** The coarse dynamics governed by χ reproduces the same swap+translation structure on a 16-state quotient.

5. **Redundancy and erasure geometry:**  
   - There is a complete map of which 4-bit erasures are reconstructable and how they affect quotient information.  
   - The information-set threshold is 8 bits.

6. **Provenance degeneracy:**  
   - With a restricted alphabet, path histories are many-to-one relative to final states, with measured conditional entropy and structured 64×64 reachable sets.

7. **Hilbert-lift entanglement behavior:**  
   - Separable code-subsets exhibit zero entropy, and bijection graphs exhibit maximal entropy, matching standard bipartite entanglement structure.

8. **Cross-domain meta-projector:**  
   - A well-defined 18D cross-domain projector arises from the K₃ geometry of domains, with exact rank and invariances.

These are properties of the discrete kernel and its associated mask code and ledgers, and they are established by the cited test suites.