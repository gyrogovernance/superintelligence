(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/olmo_forward_trace.py
OLMo Forward Pass Trace
=====

Loading model from data\models\Olmo-3-7B-Instruct...
`dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 355/355 [00:00<00:00, 3220.71it/s, Materi
Model loaded in 1.6s

-----
Model Configuration (from config.json)
-----
  hidden_size: 4096
  num_hidden_layers: 32
  num_attention_heads: 32
  num_key_value_heads: 32
  intermediate_size: 11008
  vocab_size: 100278
  sliding_window: 4096
  max_position_embeddings: 65536
  rms_norm_eps: 1e-06
  hidden_act: silu
  layer_types pattern: ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention']... (repeats)
  full_attention layers: [3, 7, 11, 15, 19, 23, 27, 31]
  sliding_attention layers: 24 layers

-----
Layer 17 Structure (peak horizon signal)
-----

  self_attn:
    type: Olmo3Attention
    q_proj:
      weight_shape: [4096, 4096]
      has_bias: False
    k_proj:
      weight_shape: [4096, 4096]
      has_bias: False
    v_proj:
      weight_shape: [4096, 4096]
      has_bias: False
    o_proj:
      weight_shape: [4096, 4096]
      has_bias: False
    q_norm:
      type: Olmo3RMSNorm
      weight_shape: [4096]
    k_norm:
      type: Olmo3RMSNorm
      weight_shape: [4096]

  post_attention_layernorm:
    type: Olmo3RMSNorm
    weight_shape: [4096]

  mlp:
    type: Olmo3MLP
    gate_proj:
      weight_shape: [11008, 4096]
      has_bias: False
    up_proj:
      weight_shape: [11008, 4096]
      has_bias: False
    down_proj:
      weight_shape: [4096, 11008]
      has_bias: False

  post_feedforward_layernorm:
    type: Olmo3RMSNorm
    weight_shape: [4096]

-----
H27@L17 Weight Analysis (horizon-tracking head)
-----
  layer: 17
  head: 27
  hidden_dim: 4096
  num_heads: 32
  head_dim: 128

  q_proj:
    full_shape: [4096, 4096]
    head_slice: [3456:3584, :]
    head_weight_shape: [128, 4096]
    head_weight_norm: 14.5625

  k_proj:
    full_shape: [4096, 4096]
    head_slice: [3456:3584, :]
    head_weight_shape: [128, 4096]
    head_weight_norm: 14.3125

  v_proj:
    full_shape: [4096, 4096]
    head_slice: [3456:3584, :]
    head_weight_shape: [128, 4096]
    head_weight_norm: 16.25

  o_proj:
    full_shape: [4096, 4096]
    head_slice: [:, 3456:3584]
    head_weight_shape: [4096, 128]
    head_weight_norm: 16.125

-----
Forward Pass Trace (short prompt)
-----
  Prompt: 'Hello world'
  Token count: 2

  Forward pass time: 1836.4ms
  Total module calls: 388
  Total weight params accessed: 7,298,011,136

  Module types called:
    Linear: 225
    Olmo3RMSNorm: 129
    SiLUActivation: 32
    Embedding: 1
    Olmo3RotaryEmbedding: 1

  First 10 Linear operations:
    1. model.layers.0.self_attn.q_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]
    2. model.layers.0.self_attn.k_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]
    3. model.layers.0.self_attn.v_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]
    4. model.layers.0.self_attn.o_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]
    5. model.layers.0.mlp.gate_proj
       Weight: [11008, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 11008]]
    6. model.layers.0.mlp.up_proj
       Weight: [11008, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 11008]]
    7. model.layers.0.mlp.down_proj
       Weight: [4096, 11008], In: [[1, 2, 11008]], Out: [[1, 2, 4096]]
    8. model.layers.1.self_attn.q_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]
    9. model.layers.1.self_attn.k_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]
    10. model.layers.1.self_attn.v_proj
       Weight: [4096, 4096], In: [[1, 2, 4096]], Out: [[1, 2, 4096]]

Forward pass trace complete.
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/torch_internals_probe.py 
PyTorch Transformer Internals Probe
=====

For Gyroscopic ASI: Understanding weight access and computation steps

-----
1. Weight Storage Analysis
-----
  weight_type: Parameter
  weight_shape: [32, 64]
  weight_dtype: torch.float32
  weight_device: cpu
  weight_requires_grad: True
  weight_is_contiguous: True
  weight_stride: [64, 1]
  weight_storage_offset: 0
  weight_numel: 2048
  weight_element_size: 4
  weight_nbytes: 8192
  bias_shape: [32]
  bias_dtype: torch.float32
  forward_equivalence: True
  forward_output_shape: [8, 32]

-----
1b. Embedding Storage Analysis
-----
  embed_weight_shape: [1000, 128]
  embed_weight_dtype: torch.float32
  lookup_equivalence: True
  lookup_output_shape: [3, 128]
  operation: indexing (weight[ids]), not matmul

-----
2. Attention Computation Steps
-----
  Step 0: input
    Shape: [1, 64, 4096]
    Note: Hidden states from previous layer/embedding
  Step 1: qkv_projection
    Op: x @ W
    Shapes: {'Q': [1, 64, 4096], 'K': [1, 64, 4096], 'V': [1, 64, 4096]}
    FLOPs: 3,221,225,472
    Note: Linear projection: dense matmul
  Step 2: reshape_multihead
    Op: view + transpose
    Shapes: {'Q': [1, 32, 64, 128], 'K': [1, 32, 64, 128], 'V': [1, 32, 64, 128]}
    Note: Reshape for parallel head computation (no FLOPs)
  Step 3: attention_scores
    Op: Q @ K.T * scale
    Shape: [1, 32, 64, 64]
    FLOPs: 16,777,216
    Note: ATTENTION BOTTLENECK: O(n^2) in sequence length
  Step 4: softmax
    Op: softmax(scores, dim=-1)
    Shape: [1, 32, 64, 64]
    Note: Normalize to probability distribution per query
  Step 5: attention_apply
    Op: attn_weights @ V
    Shape: [1, 32, 64, 128]
    FLOPs: 16,777,216
    Note: SECOND O(n^2): weighted sum of values
  Step 6: output_projection
    Op: reshape + x @ W_o
    Shape: [1, 64, 4096]
    FLOPs: 1,073,741,824
    Note: Project back to hidden dimension

  Attention Summary:
    total_flops: 4328521728
    attention_flops: 33554432
    projection_flops: 4294967296
    n_squared_ops: ['attention_scores', 'attention_apply']

-----
3. MLP (Gated FFN) Computation Steps
-----
  Step 0: input
    Shape: [1, 64, 4096]
  Step 1: gate_projection
    Op: x @ W_gate.T
    Shape: [1, 64, 11008]
    Weight: [11008, 4096]
    FLOPs: 2,885,681,152
  Step 2: up_projection
    Op: x @ W_up.T
    Shape: [1, 64, 11008]
    Weight: [11008, 4096]
    FLOPs: 2,885,681,152
  Step 3: gated_activation
    Op: silu(gate) * up
    Shape: [1, 64, 11008]
  Step 4: down_projection
    Op: intermediate @ W_down.T
    Shape: [1, 64, 4096]
    Weight: [4096, 11008]
    FLOPs: 2,885,681,152

-----
4. Summary of Observations
-----

  Weight Access Types (verified):

  embedding:
    operation: weight[token_id]
    complexity: O(1) lookup per token
    verified: Manual indexing matches module output

  linear:
    operation: x @ weight.T + bias
    complexity: O(in * out) per input vector
    verified: Manual matmul matches module output

  attention_scores:
    operation: Q @ K.T / sqrt(d_k)
    complexity: O(seq^2 * head_dim) per head
    note: This is the quadratic operation

  attention_apply:
    operation: attn_weights @ V
    complexity: O(seq^2 * head_dim) per head
    note: Second quadratic operation

  layernorm:
    operation: (x - mean) / std * gamma + beta
    complexity: O(dim) per position
    note: Element-wise, not matmul

  Architectural Facts:
    olmo_intermediate: 11008 = 256 x 43
    olmo_hidden: 4096
    olmo_heads: 32
    head_dim: 128
    quadratic_ops: ['attention_scores', 'attention_apply']

-----
5. OLMo-3-7B Architecture (from config.json)
-----

  Config:
    num_layers: 32
    num_heads: 32
    hidden_dim: 4096
    head_dim: 128
    intermediate_dim: 11008
    vocab_size: 100278
    sliding_window: 4096

  Factorizations:
    intermediate: 11008 = 256 x 43
    hidden: 4096 = 64 x 64
    heads: 32 = 4 x 8

  Weight Access Pattern:
    embedding: weight[token_id] -> (4096,)
    attention_qkv: hidden @ W.T -> (4096,) each
    attention_out: attended @ W_o.T -> (4096,)
    mlp_gate: hidden @ W_gate.T -> (11008,)
    mlp_up: hidden @ W_up.T -> (11008,)
    mlp_down: intermediate @ W_down.T -> (4096,)

Probe complete.
(.venv) PS F:\Development\superintelligence> & f:/Development/superintelligence/.venv/Scripts/python.exe f:/Development/superintelligence/research_mechanistic_interpretability/torch_weight_reader_probe.py
OLMo Weight Reader Probe
=====

Investigating HOW PyTorch reads transformer weights

-----
1. Linear Layer Weight Reading
-----
  Module output: [[1.100000023841858, 5.199999809265137, 4.300000190734863]]
  Manual output: [[1.100000023841858, 5.199999809265137, 4.300000190734863]]
  Match: True

  Computation breakdown:
    output[0] = 1.0*1.0 + 2.0*0.0 + 3.0*0.0 + 4.0*0.0 + 0.1 = 1.1  
    output[1] = 1.0*0.0 + 2.0*1.0 + 3.0*1.0 + 4.0*0.0 + 0.2 = 5.2  
    output[2] = 1.0*0.0 + 2.0*0.0 + 3.0*0.0 + 4.0*1.0 + 0.3 = 4.3  

-----
2. Attention Weight Reading
-----
  Shapes:
    input: [1, 4, 8]
    W_qkv: [24, 8]
    qkv_combined: [1, 4, 24]
    Q: [1, 4, 8]
    K: [1, 4, 8]
    V: [1, 4, 8]
  Multi-head shapes:
    Q_heads: [1, 2, 4, 4]
    K_heads: [1, 2, 4, 4]
    V_heads: [1, 2, 4, 4]
  Attention mechanics:
    operation: Compute seq x seq scores, then softmax, then apply to V
    complexity: O(seq^2) in the scores and apply steps

-----
3. MLP Weight Reading
-----

  GATE:
    input_shape: [1, 4, 16]
    weight_shape: [64, 16]
    output_shape: [1, 4, 64]
    operation: x @ W_gate.T
    weight_access: Row i of W_gate is read for output channel i    

  UP:
    input_shape: [1, 4, 16]
    weight_shape: [64, 16]
    output_shape: [1, 4, 64]
    operation: x @ W_up.T

  ACTIVATION:
    gate_shape: [1, 4, 64]
    up_shape: [1, 4, 64]
    intermediate_shape: [1, 4, 64]
    operation: silu(gate) * up (element-wise)

  DOWN:
    input_shape: [1, 4, 64]
    weight_shape: [16, 64]
    output_shape: [1, 4, 16]
    operation: intermediate @ W_down.T

  Architectural note:
    olmo_intermediate: 11008
    factorization: 256 x 43
    note: This factorization is an architectural fact, not a claim about routing

-----
4. Embedding Weight Reading
-----
  Shapes:
    token_ids: [1, 4]
    weight: [1000, 32]
    output: [1, 4, 32]
  Is indexing (not matmul): True
  Access pattern:
    operation: weight[token_id]
    for_token_42: Read row 42 of weight matrix -> 32-dim vector    
    no_matmul: This is array indexing, not matrix multiplication   

-----
5. LayerNorm Weight Reading
-----
  Shapes:
    input: [1, 4, 8]
    weight_gamma: [8]
    bias_beta: [8]
    output: [1, 4, 8]
  Weight access:
    gamma: Element-wise multiplication (not matmul)
    beta: Element-wise addition
    per_feature: Each feature has its own scale/shift

-----
6. Operations Summary
-----
  embedding: weight[token_id] (verified)
  linear: x @ weight.T + bias (verified)
  attention_scores: Q @ K.T / sqrt(d_k)
  attention_apply: attn_weights @ V
  layernorm: (x - mean) / std * gamma + beta (verified)

Probe complete.
(.venv) PS F:\Development\superintelligence> 