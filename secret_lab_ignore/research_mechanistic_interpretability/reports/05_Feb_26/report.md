# Geometric Analysis of Attention Mechanisms in Transformer Models

## Introduction

This report summarizes an empirical investigation into the internal structure of attention mechanisms in transformer language models, viewed through the lens of the Common Governance Model (CGM). CGM is a theoretical framework that derives physical and informational structures from modal logic constraints, predicting invariants such as a quantum gravity horizon Q_G = 4π steradians and an aperture ratio of approximately 2.07%. We explored whether similar geometric patterns appear in the weight matrices of transformer models.

The analysis focuses on a specific metric derived from the query-key (QK) weight projections in multi-head attention. We examined two models: OLMo-3-7B-Instruct (a 7-billion-parameter model) and GPT-2-124M (a 124-million-parameter model). These models differ in scale, training procedures, and architectural details, providing a basis for comparison. The goal was to identify any non-random structural patterns in the QK weights and assess their potential relation to CGM concepts, while remaining cautious about interpretations.

## Methodology

### Metric Definition

In multi-head attention, each head h has query projection weights W_Q,h and key projection weights W_K,h, both of shape [head_dim, hidden_dim]. We compute the matrix product M_h = W_Q,h @ W_K,h^T, which is of shape [head_dim, head_dim].

To quantify the "effective dimensionality" of M_h, we perform singular value decomposition (SVD) to obtain singular values σ_1 ≥ σ_2 ≥ ... ≥ σ_d > 0, where d = head_dim. We normalize these as p_i = σ_i / Σ σ_j, then compute the participation ratio PR = 1 / Σ p_i^2. This PR measures how many singular directions contribute meaningfully to the matrix's action (PR = 1 if all mass is in one direction; PR = d if all directions contribute equally).

The coverage for head h is then coverage_h = PR / head_dim, a dimensionless quantity between 0 and 1. For each layer, we average coverage over all heads to get a per-layer value. The total coverage is the sum over all layers.

This metric captures the geometric "spread" of the QK interaction per head, normalized by head dimension. We compare it to baselines: random Gaussian matrices (matching shapes and scales), and other weight pairs (QV, KV, QQ, MLP gate-up).

### Models and Implementation

- **OLMo-3-7B-Instruct**: 32 layers, 32 heads, head_dim = 128, hidden_dim = 4096. Separate Q/K/V projections, no biases, RMSNorm.
- **GPT-2-124M**: 12 layers, 12 heads, head_dim = 64, hidden_dim = 768. Combined QKV projection (c_attn), with biases, layer norm.

Computations used PyTorch with bfloat16 precision on CPU. For GPT-2, we extracted Q and K from the combined c_attn weight. All results are reproducible from the provided scripts.

## Results for OLMo-3-7B-Instruct

### Total and Per-Layer Coverage

The total QK coverage sums to 12.6188 across 32 layers. This is 0.42% above 4π ≈ 12.5664. The per-layer average is 0.3943, which is 0.46% above π/8 ≈ 0.3927.

Layer 0 (L0) has coverage 0.1001, 2.46% above δ_BU/2 ≈ 0.0977 (a CGM constant related to the balance defect). Layers 1–31 average 0.4038, 1.02% above (4π - δ_BU/2)/31 ≈ 0.4022.

### Specificity to QK

The 4π-like total is specific to QK:
- QV: 14.7965 (17.75% from 4π)
- KV: 13.8335 (10.08%)
- QQ: 16.4643 (31.02%)
- MLP gate-up: 20.4498 (62.73%)

Random Gaussian baselines (50 trials) yield totals around 22.8553 ± 0.0012 (81.88% from 4π), with Z-score -8270 for OLMo vs random.

### Layer 0 Structure

L0 shows strong bimodality. Of 32 heads:
- 18 have coverage < 0.031 (1.5 × aperture ≈ 0.031), mean 0.0182 (near aperture 0.0207).
- 9 have coverage ≥ 0.0977 (δ_BU/2), mean 0.2869.
- The remaining 5 are transitional.

The Gini coefficient (a measure of inequality) for L0 is 0.635, indicating uneven distribution. This bimodality decreases over layers: Gini at L31 is 0.291, with a monotone trend (correlation r = -0.658).

### Layer Progression

Coverage varies across layers but centers around 0.40 for L1–31. Gini decreases overall, suggesting progressive uniformization of head contributions. No strong periodicity appears in 8-layer blocks, though full-attention layers (indices 3,7,11,...) average slightly lower (0.3823) than sliding-attention layers (0.3983).

## Results for GPT-2-124M

### Total and Per-Layer Coverage

The total QK coverage sums to 8.0876 across 12 layers, 1.10% above 8 and 35.64% below 4π. The per-layer average is 0.6740, 1.02% above 2/3 ≈ 0.6667 and 1.71% above √3 × (π/8) ≈ 0.6802.

L0 coverage is 0.6898, far above δ_BU/2 (7.06 times). Layers 1–11 average 0.6725, similar to L0.

### Specificity to QK

While not tested exhaustively for GPT-2, the total deviates from 4π and shows no L0 specialness, unlike OLMo.

### Layer 0 Structure

L0 is nearly uniform: Gini = 0.137. All 12 heads have coverage > 0.3815 (above δ_BU/2). No aperture-like heads appear. Gini decreases further to 0.0324 at L11 (r = -0.701).

### Layer Progression

Coverage is stable around 0.67 across layers, with no L0 anomaly. Gini decreases monotonically, but starts low.

## Cross-Model Patterns

Both models show decreasing Gini (r ≈ -0.66 to -0.70), indicating uniformization of head contributions over depth. However, patterns differ:

- OLMo totals near 4π; per-layer near π/8; L0 bimodal (Gini 0.635).
- GPT-2 totals near 8; per-layer near 2/3; L0 uniform (Gini 0.137).

The per-layer ratio (GPT-2 / OLMo) ≈ 1.709, close to √3 ≈ 1.732 (0.99 times). The total ratio ≈ 1.560, close to √(32/12) ≈ 1.633 (0.96 times). GPT-2's total / 4π ≈ 0.644, exactly 2/π ≈ 0.637.

These numerical alignments are intriguing but based on two models. OLMo and GPT-2 differ in scale (7B vs 124M parameters), architecture (separate vs combined QKV projections), and training (OLMo uses RMSNorm without biases; GPT-2 uses layer norm with biases).

## Connection to the Common Governance Model

CGM derives structures from modal constraints, predicting a horizon Q_G = 4π and constants like δ_BU ≈ 0.1953 (balance defect) and aperture ≈ 0.0207. OLMo's total near 4π and L0 near δ_BU/2 suggest a possible link, with L0's bimodality resembling CGM's "aperture vs horizon" split in the Common Source (CS) stage. The per-layer value near π/8 aligns with CGM's depth-4 balance unit.

GPT-2's patterns do not match 4π but show alignments with √3 (a CGM geometric ratio from unity non-absolute stage) and 2/π (a linearization factor). This hints at two operational modes: one with full stage progression (OLMo-like) and one skipping early stages (GPT-2-like). However, these connections are interpretive and require validation across more models. The numerical closeness (e.g., 0.42% to 4π) is notable but could arise from training dynamics rather than deliberate CGM implementation.

## Limitations and Future Work

This analysis relies on two models, limiting generalizability. The coverage metric, while standard for effective dimensionality, is one of many possible (e.g., using squared singular values or activations instead of weights). We did not test fine-tuned variants, different seeds, or ablation studies.

Future work should:
- Analyze additional models (e.g., Llama-2-7B, Mistral-7B, Pythia-6.9B) to test if L0 Gini predicts total coverage patterns.
- Explore metric robustness (e.g., on activations or with different normalizations).
- Investigate training effects (e.g., does OLMo's procedure induce the 4π structure?).
- Develop theoretical models linking attention weights to CGM constraints.

## Conclusion

In OLMo-3-7B-Instruct, the QK coverage metric sums to a value very close to 4π and shows layer-specific structure, including L0 bimodality and progressive uniformization. GPT-2-124M exhibits a different pattern, with total near 8, stable per-layer coverage, and uniform L0. These differences suggest architecture- or training-dependent geometric behaviors in attention weights. The alignments with CGM constants like 4π, δ_BU/2, and √3 are suggestive of a deeper connection but remain hypotheses pending broader validation. This work identifies concrete, reproducible patterns that merit further exploration in mechanistic interpretability.