# Mechanistic Interpretability Report: OLMo-3-7B Through the CGM Lens

**Date**: 2026-02-02  
**Model**: OLMo-3-7B-Instruct  
**Framework**: Common Governance Model (CGM)  
**Kernel**: GGG ASI Alignment Router (Atlas v2.x: `ontology.npy`, `epistemology.npy`, `phenomenology.npz`)  

**Scripts used**  
- Prompt-based: `research_mechanistic_interpretability/Olmo-3-tests.py` (earlier phase)  
- Prompt-free: `research_mechanistic_interpretability/cgm_tomography.py` (current phase)  

All results in this document are empirical observations on OLMo-3-7B-Instruct under the specific probes described. We do not claim that they hold for all transformer architectures or training regimes.

---

## 0. Framing: CGM and Transformers

The Common Governance Model (CGM) is a logical and geometric framework for describing constraints on coherent recursive processes. It specifies:

- a coordinate structure (3 dimensions and 6 degrees of freedom),
- a topology (for example a K₄ tetrahedral graph and SE(3)-type motion),
- some invariants (such as Q_G = 4π, δ_BU, m_a),
- a depth structure (constraints labelled CS, UNA, ONA, BU).

Transformers are learned dynamical systems realized by tensor operations. Whatever they learn is determined by the training data and optimization procedure. CGM provides one way of asking whether certain structural motifs appear inside trained models.

This report uses the GGG Router Kernel as a fixed geometric reference. The kernel defines a finite state space Ω with:

- 65,536 ontology states (a 256 by 256 mask code product),
- a 256-byte action alphabet,
- a horizon set H of 256 states fixed by a reference byte,
- a K₄ quotient structure on masks (vertex charge χ in {0,1,2,3}),
- a Hodge decomposition on K₄ edges (gradient vs cycle, aperture).

A semantic codec maps token ids to 4 bytes. Routing these bytes through the kernel from the archetype gives each token a set of kernel coordinates:

- horizon index h(t) in {0..255},
- vertex class χ(t) in {0,1,2,3},
- mask codes u(t), v(t) in the mask code C.

Prompts then define trajectories through this discrete topology. The topology comes from the CGM-derived kernel and atlas, while trajectories are determined by the text and tokenizer.

The main question explored here is:

> Given this fixed kernel geometry, what patterns can we see in OLMo-3-7B's learned representations when we label tokens with horizon and K₄ vertex coordinates?

We do not assume that OLMo implements CGM "by design". We use CGM as a lens to organize measurements.

---

## 1. Phase I: Prompt-Based Kernel Probes

### 1.1 Initial observation: L17/H27 horizon alignment near a chat boundary

In an earlier phase, `Olmo-3-tests.py` was used with two prompts embedded in a chat template:

- a short baseline quote (36 tokens),
- a longer CS axiom text (105 tokens).

For each token position, we:

1. Routed tokens through the kernel via the semantic codec and recorded the kernel horizon label h(t),
2. Extracted attention matrices per layer and head from OLMo,
3. For a fixed query position t\*, measured how much attention mass landed on positions t with the same horizon label h(t) = h(t\*), relative to the baseline fraction p₀ of matching horizons in the sequence.

In these specific tests, the query token was the template boundary just before the assistant reply. For that token, we observed that:

- Layer 17, head 27 (L17/H27) concentrated attention almost entirely on the previous token that shared its horizon label,
- the baseline probability p₀ of a random position sharing that horizon label was about 0.0096 (roughly 1 match among 104 positions),
- the attention enrichment for same-horizon positions was on the order of 100 times, with around 0.96 of the mass on that single matching position.

A control that compared horizon-match versus token-id-match suggested that, in these prompts, the head was more sensitive to the kernel horizon label than to token identity.

This led to an early interpretation of L17/H27 as a "horizon pointer" head for this boundary token.

### 1.2 Reassessment: boundary-local effect

A closer look showed that this effect was tied to the structure of the chat template:

- the positions involved corresponded to the assistant marker and the newline token before the assistant reply,
- the codec mapping and horizon labels for these special tokens were highly structured,
- for this reason, horizon label, template role and position were not independent.

Therefore:

- the observed enrichment indicates that, in these setups, some heads can track a particular horizon label that happens to align with template structure,
- the effect is highly local to that specific boundary position and prompt format,
- it does not by itself establish a general horizon-reading mechanism across arbitrary contexts.

Phase I was useful for intuition and for checking that the kernel labels interact in nontrivial ways with actual attention patterns, but it is not taken here as strong global evidence.

This motivated a shift to a prompt-free, weight-level analysis in Phase II.

---

## 2. Phase II: Prompt-Free CGM Tomography

To reduce dependence on prompt format and particular trajectories, we moved to a prompt-free analysis in `cgm_tomography.py`. The idea is:

- sample token ids uniformly from the vocabulary,
- label each token with kernel coordinates (horizon h(t), vertex χ(t)) obtained by routing its 4-byte code through the kernel from the archetype,
- extract the corresponding embeddings from `model.embed_tokens`,
- and later, Q/K projections and head-specific projections.

This allows us to ask:

1. How much of the K₄ vertex structure is visible in the embeddings,
2. How much that structure is sharpened (or not) by Q and K projections at different layers,
3. Whether some attention heads appear more "vertex preserving" and others more "vertex mixing",
4. Whether a simple depth-4 alternation identity appears anywhere in the layerwise dynamics.

### 2.1 Data used

For the tomography run summarized here:

- `N_TOKENS = 4096` token ids were sampled uniformly from the vocabulary,
- vertices were balanced to get an effective N = 1972, with vertex counts [493, 493, 493, 493],
- in this sample, horizons covered 224 of the 256 possible values,
- the embedding dimension was 4096.

All numerical results below refer to this balanced sample unless otherwise noted.

---

## 3. T1: Quotient Visibility in Embeddings

We measured how separable kernel labels are in the embedding space using a Fisher-like class separability ratio. For two labelings:

- vertex label χ(t) in {0,1,2,3},
- horizon label h(t) in {0..255},

we computed a ratio of between-class to within-class scatter.

For the balanced sample (N = 1972):

- vertex (K₄) separability ratio ≈ 0.0026 (4 classes),
- horizon separability ratio ≈ 0.1486 (224 classes).

Within this probe design, this suggests:

- a weak but nonzero signal for the 4-way K₄ vertex labeling in the raw embedding basis,
- a much stronger signal for the 256-way horizon labeling, which is consistent with how the semantic codec was constructed (it used embeddings and thus may embed some of that structure back into the token space).

From a CGM point of view, one can say that a K₄-like quotient is detectable in the embedding space, but it is not very pronounced at this stage. Horizon information, by contrast, is much more salient in this particular setup.

---

## 4. T2: Effect of Q/K Projections on K₄ Separability

We next asked how this vertex separability changes when embeddings are projected through Q and K at different layers and heads.

For layers L in {0, 8, 16, 24, 31} and a few heads H in {0, 8, 16, 24}, we:

1. Took the Q projection matrix for that head, W_Q[L][H] of shape [head_dim × 4096],
2. computed Q_proj(t) = W_Q[L][H] · e(t) for each embedding e(t),
3. recomputed the same vertex separability ratio on the projected representations,
4. compared it to the baseline ratio from embeddings (about 0.0026).

Averaged over the sampled heads, the approximate sharpening factors were:

- Layer 0: Q ≈ 4.0 times, K ≈ 2.5 times,
- Layer 8: Q ≈ 1.15 times, K ≈ 1.33 times,
- Layer 16: Q ≈ 1.05 times, K ≈ 1.13 times,
- Layer 24: Q ≈ 1.04 times, K ≈ 1.27 times,
- Layer 31: Q ≈ 1.03 times, K ≈ 1.44 times.

Within the limitations of this simple probe, Layer 0 appears to produce the largest amplification of K₄ vertex separability through its Q (and to a lesser extent K) projections. Later layers retain this structure with modest additional changes rather than large new separations.

From a CGM perspective, it is natural to interpret this as the network amplifying a 4-way quotient structure that is weakly present in the embeddings. However, the separability remains small in absolute terms, and we do not claim that K₄ is the only or primary organizing principle.

---

## 5. T3: MLP 256×43 Structure and Horizon

OLMo-3-7B's MLP intermediate dimension is 11008 = 256 × 43. Since the kernel horizon index also ranges over 256 values, it is natural to ask if any direct horizon-channel mapping is visible.

For each layer L, we:

- took the MLP gate projection `gate(t)` in ℝ¹¹⁰⁰⁸,
- reshaped it into G(t) in ℝ²⁵⁶×⁴³,
- defined a per-channel energy E_c(t) = ||G(t)[c,:]||₂,
- asked how often the argmax_c E_c(t) equals the kernel horizon index h(t).

Across layers, we observed:

- match rates in the range 0.15% to 0.50%,
- chance level is 1/256 ≈ 0.39%,
- correlations between channel index and horizon label near zero.

Examples:

- L0: lift ≈ 0.65 times chance, correlation ≈ -0.024,
- L8: lift ≈ 1.17 times chance, correlation ≈ -0.019,
- L16: lift ≈ 0.26 times chance, correlation ≈ 0.004,
- L24: lift ≈ 1.17 times chance, correlation ≈ -0.015,
- L31: lift ≈ 1.17 times chance, correlation ≈ 0.023.

Given these numbers, we do not see evidence that the 256×43 factorization implements a simple "one horizon per channel" pattern under this probe. The dimensional match to 256 is present at the architectural level, but the functional alignment between individual channels and horizon labels does not show up clearly with this method.

---

## 6. T4: Depth-4 Alternation (XYXY vs YXYX)

CGM includes a depth-4 closure condition (BU-Egress) for certain alternating compositions. As a rough analogue in OLMo, we compared:

- the effect of the sequences [a, b] and [b, a],
- and the sequences [a, b, a, b] and [b, a, b, a],

on hidden states at different layers, using random token pairs (a, b).

For a given layer L, let F_L([seq]) be the final hidden state at that layer after feeding a sequence `seq`. We defined:

- D₂ = ||F_L([a, b]) - F_L([b, a])||₂,
- D₄ = ||F_L([a, b, a, b]) - F_L([b, a, b, a])||₂.

Averaged over 12 random pairs, for each layer we found:

- L0: D₂ ≈ 11.6, D₄ ≈ 11.5, ratio ≈ 0.99,
- L8: D₂ ≈ 13.9, D₄ ≈ 12.4, ratio ≈ 0.89,
- L16: D₂ ≈ 22.0, D₄ ≈ 21.1, ratio ≈ 0.96,
- L24: D₂ ≈ 39.0, D₄ ≈ 37.1, ratio ≈ 0.95,
- L31: D₂ ≈ 66.8, D₄ ≈ 62.5, ratio ≈ 0.94.

There is a modest reduction at L8, but in general D₄ is of the same order as D₂. No layer shows a striking regime where D₄ is dramatically smaller than D₂.

Within this limited test, we did not find a strong analogy of BU-Egress style depth-4 closure in the layerwise hidden state dynamics. This is consistent with the idea that OLMo operates in a more general, less constrained regime than the specific closure conditions defined in CGM. Stronger connections would require more targeted constructions.

---

## 7. T5: Chirality and Q/K Asymmetry

To investigate chirality-like asymmetries, we examined Q and K projection matrices and their effects.

For several layers L and for aggregated heads, we computed:

- a relative Frobenius difference `||W_Q - W_K|| / ||W_Q||`,
- an approximate commutator norm `||W_Q W_Kᵀ - W_K W_Qᵀ||` on random sketches,
- the mean cosine similarity between W_Q x and W_K x over random x.

For layers L in {0, 8, 16, 24, 31} we observed:

- relative differences ≈ 1.22 to 1.37,
- commutator scales ≈ 1.2 to 2.4 in the probed setting,
- mean cosine similarities ≈ 0.09 to 0.20.

This indicates that:

- Q and K projections are not close to each other in operator norm,
- their actions on random probes differ significantly,
- the representations they produce from the same input are almost orthogonal on average.

From a CGM viewpoint, these are consistent with a strong asymmetry between "left" and "right" style projections, similar in spirit to chirality. Here we simply record these numerical asymmetries and do not attempt to map them one-to-one onto CGM operators.

---

## 8. T6: Vertex Transport and Gradient/Cycle Heads

Finally, we looked at how attention heads move K₄ vertex labels between queries and keys.

For each layer L and head h, we built a vertex transport matrix M_h[L] from Q/K projections and vertex labels:

1. For each token in the tomography sample, we computed Q_h(t), K_h(t) in ℝ^{head_dim},
2. For each vertex label v, we formed centroids μ_Q(v) and μ_K(v) by averaging Q_h(t) and K_h(t) over tokens with χ(t) = v,
3. Defined M_h[L] = μ_Q μ_Kᵀ / √(head_dim), a 4×4 matrix,
4. From M_h[L], we derived:
   - a "diagonal advantage" (mean of diagonal minus mean of off-diagonal entries),
   - a simple antisymmetric component and its projection onto the 3D cycle space of K₄ (using the K₄ incidence matrix and its Hodge projector),
   - summarizing this as an "aperture" for that head.

Using these diagnostics on the balanced sample, we found:

- At each layer, some heads had large diagonal advantage and small "aperture". These heads tend to map each vertex strongly back to itself, and weakly to others, under this probe.
- Other heads had lower diagonal advantage and larger "aperture", indicating more mixing between different vertices.

Example patterns (illustrative, not exhaustive):

- Layer 0:
  - some heads with diag_adv around 0.50 and aperture around 0.27,
  - many heads with smaller values.
- Layer 8:
  - some heads with diag_adv around 0.58 and aperture around 0.28,
  - others with more moderate values.
- Layers 16, 24, 31:
  - similar variety, with both more "diagonal" and more "mixing" heads present.

This suggests a functional differentiation between heads when viewed through the K₄ vertex label:

- some appear more "vertex preserving" under this metric,
- others appear more "vertex mixing".

In CGM terms, these roles resemble a split between "gradient-like" heads (which preserve a notion of class) and "cycle-like" heads (which support circulation across classes). However, this is an interpretive mapping. The underlying observation is that, relative to the K₄ labels induced by the kernel, OLMo-3-7B's attention heads show a range of behaviors, some more conservative and some more mixing.

---

## 9. Synthesis

Across the probes described above on OLMo-3-7B, we can summarize the observations as follows:

- There is a weak but measurable 4-way K₄ vertex signal in the embeddings under the kernel labeling used, and this signal is amplified by Q/K projections in the first layer.
- Horizon labels are strongly visible under the same codec and kernel labeling scheme, which is consistent with how the codec was built.
- The MLP intermediate dimension 11008 = 256 × 43 aligns numerically with a 256-horizon structure, but our simple channel-horizon matching probe does not find a strong direct "one channel per horizon" pattern.
- The depth-4 alternation test ([a,b] vs [b,a] and [a,b,a,b] vs [b,a,b,a]) does not show a strong closure effect in hidden state norms at the layers we tested.
- Q and K projections differ significantly in operator norm and in how they act on random vectors, which is consistent with a strong left/right asymmetry.
- Attention heads, when restricted to the K₄ vertex labels, show a spread of behaviors: some emphasize staying within a vertex, others mix vertices more freely.

Taken together, these results do not show that OLMo "implements CGM" in a strict sense. They do indicate that, when we impose a particular K₄ and horizon labeling via the CGM-derived kernel, some of the internal structure of OLMo-3-7B can be usefully organized with respect to these labels.

The "CGM view" comments throughout are intended as one possible interpretation, not as a unique or necessary explanation. The data themselves are measurements of separability, asymmetry and transport under a chosen coordinate system.

---

## 11. Notes on Torch Internals

All experiments above rely on a standard understanding of how PyTorch transformers implement their core operations. In particular:

- Embedding lookup is implemented as `embedding[token_id]`,
- Linear layers are implemented as `x @ W.T + b`,
- Attention uses projections Q, K and V:
  - QKV projections: `hidden @ W_qkv.T`,
  - scores: `Q @ K.T / sqrt(d_k)`,
  - context: `softmax(scores) @ V`,
- MLP blocks apply gate, up and down projections with a SiLU nonlinearity.

The helper scripts:

```bash
python research_mechanistic_interpretability/torch_internals_probe.py
python research_mechanistic_interpretability/torch_weight_reader_probe.py
python research_mechanistic_interpretability/olmo_forward_trace.py