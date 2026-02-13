# PyTorch Transformer Internals Report: Weight Reading Mechanics

**Date**: 2026-02-02  
**Model**: OLMo-3-7B-Instruct  
**Purpose**: Document how PyTorch transformers read weights and execute computation  

---

## 1. OLMo-3-7B Configuration (from config.json)

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads | 32 |
| intermediate_size | 11008 |
| vocab_size | 100278 |
| sliding_window | 4096 |
| max_position_embeddings | 65536 |
| hidden_act | silu |
| rms_norm_eps | 1e-06 |
| rope_type | yarn (factor=8.0) |

### Layer Types Pattern

The model uses alternating attention types:
- Pattern: `[sliding, sliding, sliding, full]` repeated 8 times
- Full attention layers: 3, 7, 11, 15, 19, 23, 27, 31
- Sliding attention layers: all others (window=4096)

### Per-Layer Components (from weight_map and forward trace)

Each layer contains:
- `self_attn.q_proj.weight` [4096, 4096]
- `self_attn.k_proj.weight` [4096, 4096]
- `self_attn.v_proj.weight` [4096, 4096]
- `self_attn.o_proj.weight` [4096, 4096]
- `self_attn.q_norm.weight` [4096] - QK normalization (Olmo3RMSNorm)
- `self_attn.k_norm.weight` [4096] - QK normalization (Olmo3RMSNorm)
- `post_attention_layernorm.weight` [4096] (Olmo3RMSNorm)
- `mlp.gate_proj.weight` [11008, 4096]
- `mlp.up_proj.weight` [11008, 4096]
- `mlp.down_proj.weight` [4096, 11008]
- `post_feedforward_layernorm.weight` [4096] (Olmo3RMSNorm)

Total parameters: 14,596,022,272 bytes (~14.6 GB)

### H27@L17 Weight Analysis (from forward trace)

| Projection | Slice | Shape | Frobenius Norm |
|------------|-------|-------|----------------|
| q_proj | [3456:3584, :] | [128, 4096] | 14.5625 |
| k_proj | [3456:3584, :] | [128, 4096] | 14.3125 |
| v_proj | [3456:3584, :] | [128, 4096] | 16.25 |
| o_proj | [:, 3456:3584] | [4096, 128] | 16.125 |

---

## 2. Weight Access Types (Verified)

| Type | Operation | Verified |
|------|-----------|----------|
| Embedding | `weight[token_id]` | Yes - manual indexing matches |
| Linear | `x @ weight.T + bias` | Yes - manual matmul matches |
| Attention scores | `Q @ K.T / sqrt(d_k)` | Architecture |
| Attention apply | `attn_weights @ V` | Architecture |
| RMSNorm | `x * weight / rms(x)` | Architecture |

---

## 3. Forward Pass Operations (from tracer)

### Module calls per forward pass (2-token prompt: "Hello world")

| Module Type | Count |
|-------------|-------|
| Linear | 225 |
| Olmo3RMSNorm | 129 |
| SiLUActivation | 32 |
| Embedding | 1 |
| Olmo3RotaryEmbedding | 1 |
| **Total** | 388 |

- Total weight parameters accessed: 7,298,011,136
- Forward pass time: 1848.1ms (CPU, bfloat16)

### Per-layer operation sequence

1. Q projection: `hidden @ W_q.T`
2. K projection: `hidden @ W_k.T`
3. V projection: `hidden @ W_v.T`
4. Q/K normalization (QK-norm)
5. Rotary position embedding
6. Attention scores: `Q @ K.T / sqrt(d_k)`
7. Softmax
8. Attention apply: `attn_weights @ V`
9. O projection: `attn_out @ W_o.T`
10. Residual + post_attention_layernorm
11. Gate projection: `hidden @ W_gate.T`
12. Up projection: `hidden @ W_up.T`
13. SiLU activation + multiply
14. Down projection: `intermediate @ W_down.T`
15. Residual + post_feedforward_layernorm

---

## 4. FLOPs per Layer (seq=64, batch=1)

| Operation | FLOPs |
|-----------|-------|
| QKV projection (3x) | 3.2B |
| Attention scores | 16.7M |
| Attention apply | 16.7M |
| O projection | 1.1B |
| MLP gate | 2.9B |
| MLP up | 2.9B |
| MLP down | 2.9B |

At short sequences, projections dominate. At long sequences, attention scores (O(seq^2)) dominates.

---

## 5. Architectural Facts

- Intermediate dimension: 11008 = 256 x 43
- Hidden dimension: 4096 = 64 x 64
- Head dimension: 128 = 4096 / 32 heads
- Heads: 32 = 4 x 8
- QK normalization: Olmo3RMSNorm applied to Q and K before attention
- Two RMSNorm per layer: post_attention_layernorm, post_feedforward_layernorm
- No bias in any linear layers (has_bias: False)
- Activation: SiLU (32 calls per forward pass = 1 per layer)

---

## 6. Test Scripts

| Script | Purpose |
|--------|---------|
| `torch_internals_probe.py` | Weight storage, attention/MLP step traces |
| `torch_weight_reader_probe.py` | Verifies exact weight reading operations |
| `olmo_forward_trace.py` | Hooks into OLMo model, traces forward pass |

### Usage

```bash
python research_mechanistic_interpretability/torch_internals_probe.py
python research_mechanistic_interpretability/torch_weight_reader_probe.py
python research_mechanistic_interpretability/olmo_forward_trace.py
```
