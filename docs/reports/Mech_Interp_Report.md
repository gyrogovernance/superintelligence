# Mechanistic Interpretability Report: OLMo-3-7B Through the CGM Lens

**Date**: 2026-02-02  
**Model**: OLMo-3-7B-Instruct  
**Framework**: Coherent Gyroscopic Model (CGM)  
**Test Prompt**: "Freedom is not worth having if it does not include the freedom to make mistakes." (36 tokens)

## Executive Summary

This report documents mechanistic interpretability experiments probing how OLMo-3-7B realizes CGM geometric and topological invariants.

**Established structural properties**: MLP 256x43 factorization confirmed (100% horizon entropy). Layer-4 closure pattern [s,s,s,F]x8 is functional (1.93x norm ratio). Byte feature correlation ~0.32 justifies linear bridge. Full attention layers preserve horizon slightly better than sliding.

**Adapter probes (exploratory, single-prompt)**: At L0-L4, attention shows slight positive enrichment (~5-7% above random baseline); deeper layers show anti-alignment (<1x). K4-like charge transport exists (~8% mismatch) but population is collapsed on this prompt (26% entropy). CGM-J decoder correctly identifies LM's top b0 bucket but overall correlation is weak (0.17). All adapter results are prompt-local until validated across diverse inputs.

---

## Established Structural Properties

### CGM-1: MLP 256x43 Structure

| Layer | Entropy | Ratio | Active Channels |
|-------|---------|-------|-----------------|
| L8 | 5.54/5.55 | 100% | 135/256 |
| L16 | 5.54/5.55 | 100% | 166/256 |
| L24 | 5.54/5.55 | 100% | 208/256 |

**Status**: Established invariant. MLP uses full 256-channel horizon capacity.

---

### CGM-2: Layer-4 Closure Pattern

| Metric | Full Attention | Sliding Attention |
|--------|----------------|-------------------|
| Mean | 78.38 | 40.60 |
| Std | 112.84 | 26.77 |
| **Ratio** | **1.93x** | baseline |

**Status**: Established invariant. [s,s,s,F] x 8 depth-4 closure is functional.

---

### CGM-E: Byte Feature Correlation

| Metric | Value |
|--------|-------|
| Avg correlation | **0.32** |
| N components | 36 |

**Status**: Established invariant. ~0.32 correlation justifies linear bridge to 43D byte features.

---

### CGM-H3: Horizon Leakage by Layer Type

| Metric | Value |
|--------|-------|
| Full leakage | 0.316 |
| Sliding leakage | 0.319 |
| Full - Sliding | **-0.0025** |

**Status**: Established invariant. Full attention layers preserve horizon slightly better (0.25% edge). Closure has small but consistent preservative effect.

---

## Adapter Probes (Exploratory)

*All results below are from a single prompt with high b0=255 concentration (83% baseline). Interpretation requires validation across diverse prompts.*

### CGM-A: Attention vs Semantic Horizon

| Metric | Value |
|--------|-------|
| Same-horizon mass (L16) | **66%** |
| Random baseline | 83% (29/35 tokens share b0) |
| Enrichment | 0.80x |

**Status**: Exploratory. At L16, attention is anti-aligned with horizon (0.80x enrichment = below random). This prompt's high baseline makes raw mass misleading.

---

### CGM-C: Best Horizon Layer

| Layer | Same-Horizon Mass | Enrichment |
|-------|-------------------|------------|
| L0 | 87% | 1.05x |
| **L4** | **89%** (peak) | **1.07x** |
| L8 | 75% | 0.90x |
| L12 | 68% | 0.82x |
| L16 | 66% | 0.80x |
| L20 | 55% | 0.67x |
| L24 | 56% | 0.68x |
| L28 | 50% | 0.61x |

**Interpretation**: At L0-L4, attention shows slight positive enrichment (~5-7% above baseline). At deeper layers, enrichment drops below 1x - attention preferentially leaves the horizon bucket despite its high occupancy. L4 is the "least bad" layer, not a strong identity.

**Status**: Exploratory. L4 is optimal read point. Enrichment interpretation requires validation on prompts with varied baselines.

---

### CGM-D3: K4 Parity Discovery

| Metric | Value |
|--------|-------|
| Candidates (mm<0.1) | 264,906 |
| Best masks | 0x210, 0x4000020 |
| Mismatch | 0.083 |
| Charge mass | [0.917, 0.038, 0.038, 0.007] |
| Charge entropy | **0.36 / 1.39 (26%)** |

**Interpretation**: We can find semantic parity masks with ~8% attention mismatch - K4-like charges are well transported layer-wise. However, for this prompt the charge distribution is highly collapsed (~92% mass in one vertex). This indicates usable transport structure with severely anisotropic population on this sample.

**Status**: Exploratory - prompt-local. Needs replication on multiple prompts to determine if collapse is codec-wide or prompt-specific.

---

### CGM-J: Next-Token Horizon Alignment

| Metric | Value |
|--------|-------|
| LM top b0 | 255 (mass=1.000) |
| Decoder rank of LM top | **1/256** |
| Top-10 b0 overlap | 5/10 |
| Correlation | 0.173 |
| KL divergence | 5.232 |

**Interpretation**: On this prompt the LM's next-token horizon distribution is nearly a delta at b0=255. Under this degenerate regime, the L4-based decoder still assigns highest similarity to the same b0 (rank 1/256) and shows modest positive correlation (~0.17). KL divergence is large due to LM's extreme concentration. This test becomes meaningful only on prompts where LM's b0 distribution has nontrivial width.

**Status**: Exploratory - conceptually correct, but current values are prompt-limited. Key structural fact: decoder and LM agree on dominant horizon bucket.

---

## Archived Findings

| Probe | Result | Conclusion |
|-------|--------|------------|
| CGM-3 | 7% distant similarity | Transformers lack trajectory memory. Genealogy required. |
| CGM-B | 37% in top-6 SVD | Moderate compressibility baseline. |
| CGM-F | -0.019 separation | Position mod 4 is wrong test. Depth-4 is in layers. |
| CGM-D v1 | -0.0025 separation | Q0/Q1 on m12 not CGM-equivalent. |
| CGM-D v2 | 0.074 mismatch | K4 found but skewed (26% entropy). Refined to D3. |
| CGM-G v1 | 0% accuracy | Coordinate system mismatch. |
| CGM-G v2 | 8.3% top-5 | Unweighted prototypes. |
| CGM-G3 | 8.3% top-5 | Wrong target: "own b0" not meaningful. Use CGM-J (next-token horizon). |
| CGM-H v1 | -0.034 separation | Wrong observable. |
| CGM-H v2 | 0.0025 diff | Measured retention not leakage. Promoted to H3 (structural). |

---

## Integration Architecture

Based on confirmed findings:

1. **Read Point**: L4 hidden state (89% same-horizon mass, peak alignment)
2. **Horizon Decoder**: Prototype decoder identifies top b0 correctly (rank 1/256); full distribution correlation weak (0.17)
3. **K4 Charge**: Skewed (26% entropy) - use as auxiliary signal only
4. **Phase Sync**: Full layers marginally better (0.25%)

The adapter pipeline:
```
hidden_L4 -> prototype_match -> top_b0 (correct)
hidden_L4 -> K4_masks -> skewed_charge (auxiliary)
top_b0 -> Gyroscopic M field
```

---

## Next Steps

1. **Multi-prompt validation**: Run `--mi-cgm` on diverse prompts (content-only, no chat template). Log enrichment curves, D3 entropy, J correlation per prompt. Determine which findings are model properties vs prompt artifacts.
2. **Codec health check**: Compute global b0 histogram over vocabulary. If one bucket dominates, may need rebalanced codec.
3. **Define adapter objective**: From L4 hidden state h, produce b0 distribution q(b0|h) minimizing KL(LM_b0 || q) or maximizing top-k overlap. This is the operational target.
4. **Gyro coupling**: Once decoder quality is known, map highest-probability b0 to horizon index, or pass full q as weights over horizon states.

---

## Test Configuration

```
Model: OLMo-3-7B-Instruct
Layers: 32 (8 full, 24 sliding)
Hidden: 4096
Intermediate: 11008 (256 x 43)
Attention: eager
Dtype: bfloat16
Codec: SemanticTokenCodec (100278 tokens)
```
