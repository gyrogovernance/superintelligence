**NOTE (2025): This document is partially superseded.** The implementation has diverged:

- **AgentBlock** replaced by **DirectionalAgentBlock** (asymmetric cross-attention at u1, u3, u5)
- **TensorBlock** no longer uses a 12x12 grid mask; topology comes from L3 state injection only
- **HeadAgent** uses vectorized 6+2 (family + micro) decomposition with permutation buffer
- Ontology/epistemology mapping references may be outdated

Use `blocks.py`, `model.py`, `head.py` as source of truth.

---

Here is the complete codebase structure, grounded in everything we discussed.

---

## File Tree

```
secret_lab_ignore/autobots/
│
├── __init__.py
│
├── physics.py                  # Lossless physics (the 1KB of real information)
├── config.py                   # HGTConfig (physics constants + model hyperparams)
├── tokenizer.py                # Byte tokenizer with GENE_Mic decomposition
│
├── embeddings.py               # BL1 + TL1 (base embeddings)
├── blocks.py                   # ByteBlock, TensorBlock, DirectionalAgentBlock, TransitionBlock
├── head.py                     # HeadAgent (L4 closure → 256 logits)
├── model.py                    # HGTForCausalLM (assembles everything)
│
├── curriculum.py               # FSM trajectory generator (training only)
├── train.py                    # Training entry point
├── convert.py                  # Structural projector for external weights
│
└── tests/
    ├── __init__.py
    ├── test_physics.py         # Verify physics matches src.router.constants
    ├── test_model.py           # Forward pass, gradient flow, output shape
    ├── test_lossless.py        # Verify physics survives training unchanged
    ├── test_curriculum.py      # Verify trajectory generation covers state space
    └── test_convert.py         # Verify projection exactness on Bolmo weights
```

**Output artifacts** (generated, not committed):

```
data/autobots/
├── curriculum/                 # Generated FSM trajectories (.bin files)
├── checkpoints/                # Training checkpoints
└── model/                      # Final trained model
    ├── config.json             # Physics constants + architecture
    ├── model.safetensors       # ~16MB learned weights
    └── tokenizer_config.json   # Byte tokenizer config
```

---

## File Descriptions

### `physics.py` — The Lossless Core

This file contains **zero neural network code**. It is pure integer arithmetic that computes exact FSM states from a byte sequence. At inference time, the model calls these functions in its forward pass. They are never approximated.

```python
"""
Lossless physics computations for the HGT forward pass.

These functions compute exact FSM states using the function face
(bitwise operations). No tables needed. No approximation.

Total information content: 384 bytes (256 masks × 12 bits).
Everything else is derived from GENE_MIC_S = 0xAA.
"""

# Constants (duplicated from src.router.constants for deployment independence)
GENE_MIC_S: int = 0xAA
Q0: int = 0x033
Q1: int = 0x0F0
ARCHETYPE_A12: int = 0xAAA
ARCHETYPE_B12: int = 0x555

def intron(byte_val: int) -> int: ...
def expand_intron_to_mask12(intron: int) -> int: ...
def vertex_charge(mask12: int) -> int: ...

# Trajectory computers (sequential, no tables)
def compute_l1_trajectory(introns: Tensor) -> Tensor: ...
    """Running XOR: l1[t] = l1[t-1] ^ intron[t]. Prefix-XOR scan."""

def compute_l2_trajectory(introns: Tensor) -> Tensor: ...
    """Chirality pairs: (A8,B8) with gyration rule. Returns [batch, seq, 2]."""

def compute_l3_trajectory(introns: Tensor) -> Tensor: ...
    """Full 24-bit: (A12,B12) with mask expansion + gyration. Returns [batch, seq, 2]."""

def compute_l4_commitments(mask12s: Tensor) -> tuple[Tensor, Tensor]: ...
    """Running O/E: O = XOR of masks at even positions, E at odd. Returns (O, E)."""

def compute_mask12_table() -> Tensor: ...
    """Frozen buffer: 256 int32 values. The ONLY precomputed artifact (384 bytes)."""
```

**Dependencies**: `torch` (for Tensor operations). No `src.*` imports at runtime.
**Size**: ~200 lines.

---

### `config.py` — Model Configuration

```python
"""
HGTConfig: physics constants + model hyperparameters.

The physics constants are exact integers. They are NOT learned.
They live in config.json alongside standard transformer hyperparams.
"""

class HGTConfig(PretrainedConfig):
    model_type = "hgt"
    
    # Physics constants (exact, from the kernel specification)
    gene_mic_s: int = 0xAA
    q0: int = 0x033
    q1: int = 0x0F0
    archetype_state24: int = 0xAAA555
    
    # Vocabulary (byte-native, no BPE)
    vocab_size: int = 256
    family_size: int = 4
    micro_ref_size: int = 64
    
    # Architecture (3 resolutions matching L1/L2/L3)
    resolution_dims: tuple = (64, 128, 256)   # hidden dims for res 1/2/3
    num_heads: tuple = (4, 4, 8)              # attention heads per resolution
    ffn_multiplier: int = 4                    # FFN hidden = dim × multiplier
    
    # Sequence
    max_position_embeddings: int = 2048
```

**Dependencies**: `transformers.PretrainedConfig`.
**Size**: ~60 lines.

---

### `tokenizer.py` — Physics Tokenizer

```python
"""
Byte tokenizer that computes GENE_Mic decomposition.

Input: raw text (UTF-8 string)
Output: dict with input_ids (bytes), families, micro_refs, introns

This is where BL1 begins. The tokenizer IS the first Byte Layer's
input preparation. No BPE. No learned vocabulary. Pure physics.

Can optionally wrap BolmoTokenizer for compatibility with existing
models during the conversion phase.
"""

class PhysicsTokenizer(PreTrainedTokenizer):
    vocab_size = 256  # byte-native
    
    def encode_with_physics(self, text: str) -> dict:
        """Returns input_ids + all GENE_Mic priors per byte."""
        
    def decode(self, ids: list[int]) -> str:
        """Bytes → UTF-8 string."""
        
    # Bolmo compatibility (for converter)
    def from_bolmo_ids(self, bolmo_ids: list[int]) -> dict:
        """Convert Bolmo token IDs to physics-encoded bytes."""
```

**Dependencies**: `transformers.PreTrainedTokenizer`, `physics.py`.
**Size**: ~120 lines.

---

### `embeddings.py` — BL1 + TL1

This is where bytes enter the neural network. BL1 embeds the GENE_Mic decomposition. TL1 embeds the L1 FSM state and vertex charge. These are the only `nn.Embedding` tables in the model.

```python
"""
Base embeddings: BL1 (Byte Layer 1) and TL1 (Tensor Layer 1).

BL1 embeds the GENE_Mic decomposition:
  - byte value (0-255) → byte_embed
  - family (0-3) → family_embed  
  - micro_ref (0-63) → micro_embed
  Output: sum of three embeddings, dim = resolution_dims[0]

TL1 embeds the GENE_Mac L1 state:
  - l1_state (0-255) → l1_state_embed
  - vertex_charge (0-3) → vertex_embed
  Output: sum of two embeddings, dim = resolution_dims[0]

Both BL1 and TL1 output tensors of shape [batch, seq, dim_0].
The L4 commitments (O, E) are added as a position encoding.
"""

class ByteLayer1(nn.Module):
    """BL1: GENE_Mic embedding."""
    def __init__(self, config: HGTConfig):
        self.byte_embed = nn.Embedding(256, config.resolution_dims[0] // 2)
        self.family_embed = nn.Embedding(4, config.resolution_dims[0] // 4)
        self.micro_embed = nn.Embedding(64, config.resolution_dims[0] // 4)
    
    def forward(self, input_ids, families, micro_refs) -> Tensor: ...

class TensorLayer1(nn.Module):
    """TL1: GENE_Mac L1 state embedding."""
    def __init__(self, config: HGTConfig):
        self.l1_state_embed = nn.Embedding(256, config.resolution_dims[0] - 4)
        self.vertex_embed = nn.Embedding(4, 4)
    
    def forward(self, l1_states, vertex_charges) -> Tensor: ...

class L4PositionEncoding(nn.Module):
    """
    Projects L4 commitments (O, E) into position vectors.
    
    O and E are 12-bit integers. They are expanded to 24 binary
    features and projected to hidden_dim. The projection is LEARNED
    but the binary features are EXACT (computed by physics.py).
    """
    def __init__(self, config: HGTConfig):
        self.projection = nn.Linear(24, config.resolution_dims[0])
    
    def forward(self, l4_O, l4_E) -> Tensor: ...
```

**Dependencies**: `torch.nn`, `config.py`.
**Size**: ~150 lines.

---

### `blocks.py` — The Building Blocks

Four block types, all built from standard PyTorch components.

```python
"""
Building blocks for the HGT architecture.

ByteBlock: Self-attention on the byte stream (BL2, BL3).
TensorBlock: Self-attention on the tensor stream (TL2, TL3).
AgentBlock: Cross-attention between byte and tensor streams.
TransitionBlock: Resolution transition (8→16→24 bit).

All blocks use standard nn.MultiheadAttention and nn.Linear.
The ONLY non-standard element is the structured attention mask
in TensorBlock, which follows the 2×3×2 grid topology.
"""

class ByteBlock(nn.Module):
    """
    Self-attention on byte stream.
    Standard TransformerEncoderLayer (self-attn + FFN + LayerNorm).
    Operates on the GENE_Mic representation.
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int): ...
    def forward(self, bl: Tensor) -> Tensor: ...

class TensorBlock(nn.Module):
    """
    Self-attention on tensor stream with 2×3×2 topology.
    
    The attention mask encodes the grid structure:
    - Frame attention: bits 0-5 ↔ bits 6-11
    - Row attention: X(0,1) ↔ Y(2,3) ↔ Z(4,5) within each frame
    - Column attention: neg(0,2,4) ↔ pos(1,3,5) within each frame
    
    These masks are FROZEN BUFFERS derived from the physics,
    registered via register_buffer (never learned).
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int): ...
    def forward(self, tl: Tensor) -> Tensor: ...

class AgentBlock(nn.Module):
    """
    Cross-attention bridge between byte and tensor streams.
    
    This is a Micro Agent. It performs BIDIRECTIONAL cross-attention:
    1. bl_updated = CrossAttn(query=bl, key=tl, value=tl)
    2. tl_updated = CrossAttn(query=tl, key=bl, value=bl)
    
    Both streams are updated and returned.
    """
    def __init__(self, dim: int, num_heads: int): ...
    def forward(self, bl: Tensor, tl: Tensor) -> tuple[Tensor, Tensor]: ...

class TransitionBlock(nn.Module):
    """
    Resolution transition between FSM levels.
    
    BTL1_2: Projects from resolution_dims[0] → resolution_dims[1]
            Injects L2 state features (A8, B8 chirality pair)
    BTL2_3: Projects from resolution_dims[1] → resolution_dims[2]
            Injects L3 state features (A12, B12 topology)
    
    The injected features are EXACT (computed by physics.py).
    The projection weights are LEARNED.
    """
    def __init__(self, dim_in: int, dim_out: int, state_features: int): ...
    def forward(self, bl: Tensor, tl: Tensor, 
                state_features: Tensor) -> tuple[Tensor, Tensor]: ...
```

**Dependencies**: `torch.nn`, `config.py`.
**Size**: ~300 lines.

---

### `head.py` — The Head Agent

```python
"""
Head Agent: L4 closure → next byte prediction.

Input: final BL3 and TL3 representations + L4 commitments.
Output: 256 logits (one per byte).

The Head receives the fully entangled byte/tensor representation
and predicts the next structural mutation (mask12), which maps
1:1 to the 256-byte alphabet.

L4 commitments (O, E) enter here as conditioning — they tell the
Head where we are in the holographic closure cycle.

Architecture: concatenate BL3 + TL3 + L4_projection → MLP → 256 logits.
"""

class HeadAgent(nn.Module):
    def __init__(self, config: HGTConfig):
        self.l4_proj = nn.Linear(24, config.resolution_dims[2])
        self.combine = nn.Linear(config.resolution_dims[2] * 3, config.resolution_dims[2])
        self.output = nn.Linear(config.resolution_dims[2], 256, bias=False)
        
        # Structured initialization: group output rows by Family
        self._init_output_by_family(config)
    
    def forward(self, bl3, tl3, l4_O, l4_E) -> Tensor: ...
    
    def _init_output_by_family(self, config): ...
        """Initialize output weight rows grouped by family structure."""
```

**Dependencies**: `torch.nn`, `config.py`, `physics.py`.
**Size**: ~100 lines.

---

### `model.py` — HGTForCausalLM

The main model class. Assembles all components. Inherits from `PreTrainedModel` for HuggingFace compatibility.

```python
"""
Holographic Grid Transformer for Causal Language Modeling.

Architecture:
  Physics → BL1/TL1 → Agent μ1 → BTL1_2 → BL2/TL2 → Agent μ3 → 
  BTL2_3 → BL3/TL3 → Agent μ5 → HeadAgent → 256 logits

The forward pass has two phases:
  1. PHYSICS PHASE: Compute exact FSM states from input bytes.
     This is lossless, non-differentiable, and fast (bitwise ops).
  2. NEURAL PHASE: Embed, attend, predict.
     This is learned, differentiable, and standard PyTorch.

The physics phase outputs are DETACHED from the computation graph.
The neural phase learns the stochastic distribution OVER the 
physics-computed structural features.
"""

class HGTForCausalLM(PreTrainedModel):
    config_class = HGTConfig
    
    def __init__(self, config: HGTConfig):
        super().__init__(config)
        
        # Frozen physics buffer
        self.register_buffer('mask12_table', compute_mask12_table())
        
        # BL1 / TL1 (base embeddings)
        self.bl1 = ByteLayer1(config)
        self.tl1 = TensorLayer1(config)
        self.l4_pos = L4PositionEncoding(config)
        
        # Agent μ1: BL1 ↔ TL1
        self.agent_1 = AgentBlock(config.resolution_dims[0], config.num_heads[0])
        
        # Transition BTL1_2: 8-bit → 16-bit
        self.transition_1_2 = TransitionBlock(
            config.resolution_dims[0], config.resolution_dims[1], 
            state_features=2  # A8, B8
        )
        
        # BL2 / TL2 + Agent μ3
        self.bl2 = ByteBlock(config.resolution_dims[1], config.num_heads[1], ...)
        self.tl2 = TensorBlock(config.resolution_dims[1], config.num_heads[1], ...)
        self.agent_3 = AgentBlock(config.resolution_dims[1], config.num_heads[1])
        
        # Transition BTL2_3: 16-bit → 24-bit
        self.transition_2_3 = TransitionBlock(
            config.resolution_dims[1], config.resolution_dims[2],
            state_features=4  # A12_frame0, A12_frame1, B12_frame0, B12_frame1
        )
        
        # BL3 / TL3 + Agent μ5
        self.bl3 = ByteBlock(config.resolution_dims[2], config.num_heads[2], ...)
        self.tl3 = TensorBlock(config.resolution_dims[2], config.num_heads[2], ...)
        self.agent_5 = AgentBlock(config.resolution_dims[2], config.num_heads[2])
        
        # Head Agent
        self.head = HeadAgent(config)
    
    def forward(self, input_ids, labels=None):
        # ====== PHYSICS PHASE (exact, detached) ======
        with torch.no_grad():
            introns = input_ids ^ self.config.gene_mic_s
            families = (introns >> 6) & 0x3
            micro_refs = introns & 0x3F
            mask12s = self.mask12_table[input_ids]
            
            l1_states = compute_l1_trajectory(introns)
            vertices = compute_vertex_batch(mask12s, self.config.q0, self.config.q1)
            l2_a8, l2_b8 = compute_l2_trajectory(introns)
            l3_a12, l3_b12 = compute_l3_trajectory(introns, mask12s)
            l4_O, l4_E = compute_l4_commitments(mask12s)
        
        # ====== NEURAL PHASE (learned) ======
        # Resolution 1 (8-bit)
        bl = self.bl1(input_ids, families, micro_refs)
        tl = self.tl1(l1_states, vertices)
        bl = bl + self.l4_pos(l4_O, l4_E)
        tl = tl + self.l4_pos(l4_O, l4_E)
        bl, tl = self.agent_1(bl, tl)
        
        # Resolution 2 (16-bit)
        l2_features = torch.stack([l2_a8, l2_b8], dim=-1)
        bl, tl = self.transition_1_2(bl, tl, l2_features)
        bl = self.bl2(bl)
        tl = self.tl2(tl)
        bl, tl = self.agent_3(bl, tl)
        
        # Resolution 3 (24-bit)
        l3_features = torch.stack([
            l3_a12 & 0x3F, (l3_a12 >> 6) & 0x3F,
            l3_b12 & 0x3F, (l3_b12 >> 6) & 0x3F
        ], dim=-1)
        bl, tl = self.transition_2_3(bl, tl, l3_features)
        bl = self.bl3(bl)
        tl = self.tl3(tl)
        bl, tl = self.agent_5(bl, tl)
        
        # Head
        logits = self.head(bl, tl, l4_O, l4_E)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 256), labels.view(-1))
        
        return CausalLMOutput(loss=loss, logits=logits)
```

**Dependencies**: All autobots modules, `transformers.PreTrainedModel`.
**Size**: ~250 lines.

---

### `curriculum.py` — FSM Training Data Generator

This is the **only file** that touches the 13GB L3 table. It imports from `src.tools.layers`. It generates training data and writes it to disk. After training, it is never needed again.

```python
"""
FSM curriculum generator.

Generates training trajectories by walking the L1/L2/L3 FSMs.
This is NOT text prediction training — it is structural learning.

The generator produces byte sequences WITH full physics annotations:
  - input bytes
  - L1/L2/L3 states at each step
  - L4 commitments at each step
  - vertex charges, family transitions

The model learns to predict the next byte given the structural context.
Because the FSM is finite and complete, the curriculum can guarantee
coverage of all 256 bytes, all 4 families, all 4 vertex charges,
and a representative sample of state transitions.

IMPORTANT: This file imports from src.tools.layers (the 13GB table).
           It runs ONLY during training data generation.
           It is NOT needed at inference.
"""

from src.tools.layers import (
    Layer1FSM, Layer2FSM, Layer3FSM, Layer4,
    create_default_four_layers,
)

class FSMCurriculum:
    def __init__(self, l3_path: Path):
        self.four = create_default_four_layers(l3_path=l3_path)
    
    def generate_random_walks(self, num_sequences, seq_len) -> list: ...
        """Random walks from random starting states."""
    
    def generate_family_balanced(self, num_sequences, seq_len) -> list: ...
        """Walks that balance all 4 families equally."""
    
    def generate_closure_walks(self, num_sequences) -> list: ...
        """Walks that include depth-4 closure cycles (xyxy patterns)."""
    
    def generate_full_coverage(self) -> list: ...
        """Ensure every byte from every vertex charge is represented."""
    
    def save_curriculum(self, output_dir: Path) -> None: ...
        """Write trajectories to disk as .bin + .json metadata."""
    
    def load_curriculum(self, input_dir: Path) -> Dataset: ...
        """Load as a standard PyTorch Dataset."""
```

**Dependencies**: `src.tools.layers`, `torch.utils.data.Dataset`.
**Size**: ~200 lines.

---

### `train.py` — Training Entry Point

```python
"""
Training script for HGT.

Phase 1: Generate curriculum from FSM (requires L3 table)
Phase 2: Train model on curriculum (standard PyTorch training loop)
Phase 3: Verify lossless physics preservation
Phase 4: Save model as HuggingFace-compatible checkpoint

Usage:
    python -m secret_lab_ignore.autobots.train \
        --l3-path data/layers/l3_packed_u24.bin \
        --output-dir data/autobots/model \
        --epochs 10
"""

def main():
    # 1. Generate or load curriculum
    curriculum = FSMCurriculum(args.l3_path)
    train_dataset = curriculum.generate_full_coverage()
    
    # 2. Create model
    config = HGTConfig()
    model = HGTForCausalLM(config)
    
    # 3. Standard PyTorch training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        for batch in DataLoader(train_dataset, batch_size=32):
            loss = model(batch['input_ids'], labels=batch['labels']).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # 4. Verify physics is lossless
    verify_physics_preserved(model)
    
    # 5. Save
    model.save_pretrained(args.output_dir)
    tokenizer = PhysicsTokenizer()
    tokenizer.save_pretrained(args.output_dir)
```

**Dependencies**: All autobots modules.
**Size**: ~150 lines.

---

### `convert.py` — Structural Projector

```python
"""
Structural projection of external model weights.

Takes a trained HGT physics model and an external model (e.g. Bolmo-1B).
Decomposes external embeddings into:
  E_original = E_structural + E_residual

where E_structural is the component aligned with byte physics.

This is NOT training. It is a deterministic matrix operation.

Usage:
    python -m secret_lab_ignore.autobots.convert \
        --physics-model data/autobots/model \
        --target-model data/models/Bolmo-1B \
        --output data/autobots/bolmo_structural
"""

class StructuralProjector:
    def __init__(self, physics_model: HGTForCausalLM):
        self.physics = physics_model
    
    def compute_structural_basis(self) -> Tensor: ...
        """256 byte embeddings from the physics model → orthonormal basis."""
    
    def project_token_embeddings(self, 
            target_embeddings: Tensor, 
            target_tokenizer) -> tuple[Tensor, Tensor]: ...
        """Returns (E_structural, E_residual)."""
    
    def apply_to_model(self, target_model_path: Path, 
                       output_path: Path) -> dict: ...
        """Full conversion pipeline. Returns alignment metrics."""
```

**Dependencies**: `model.py`, `physics.py`, `torch`, `safetensors`.
**Size**: ~200 lines.

---

## Data Flow

### Training
```
src/tools/layers.py (L1 64KB, L2 32MB, L3 13GB from disk)
         │
         ▼
curriculum.py ──generates──▶ data/autobots/curriculum/*.bin
         │
         ▼
train.py ──loads curriculum──▶ model.py forward/backward
         │
         ▼
data/autobots/model/
    ├── config.json           (~200 bytes, physics constants)
    ├── model.safetensors     (~16 MB, learned weights)
    └── tokenizer_config.json (~500 bytes)
```

### Inference (No FSM tables, no src.tools dependency)
```
User text ──▶ tokenizer.py ──▶ bytes (0-255)
                                    │
                     ┌──────────────┤
                     ▼              ▼
               physics.py      model.py
            (exact bitwise)  (neural forward)
                     │              │
                     └──────┬───────┘
                            ▼
                      256 logits → next byte
```

### Conversion (Deterministic, no training)
```
data/autobots/model/ ──▶ convert.py ◀── data/models/Bolmo-1B/
                              │
                              ▼
                    data/autobots/bolmo_structural/
                        ├── config.json
                        ├── model.safetensors  (same size as Bolmo)
                        └── alignment_report.json
```

---

## Parameter Count Estimate

| Component | Params | Notes |
|-----------|--------|-------|
| BL1 embeddings | 256×32 + 4×16 + 64×16 = 9,280 | Byte + Family + Micro |
| TL1 embeddings | 256×60 + 4×4 = 15,376 | L1 state + Vertex |
| L4 position | 24×64 = 1,536 | O/E → hidden |
| Agent μ1 | 64×64×4 = 16,384 | Cross-attention |
| Transition BTL1_2 | 64×128×2 + 2×128 = 16,640 | Projection + state injection |
| ByteBlock2 + TensorBlock2 | ~660K | Two standard encoder layers |
| Agent μ3 | ~66K | Cross-attention at dim=128 |
| Transition BTL2_3 | ~66K | Projection + state injection |
| ByteBlock3 + TensorBlock3 | ~2.6M | Two standard encoder layers |
| Agent μ5 | ~262K | Cross-attention at dim=256 |
| HeadAgent | ~200K | L4 proj + combine + output |
| **Total** | **~3.9M params** | **~16 MB safetensors** |

The 13GB L3 table generates the curriculum. The deployed model is 16 MB.

===

Phase 2

Here is the complete specification of missing components for a **production-grade, theoretically faithful** implementation. Pass this checklist to your code assistant.

---

## **CRITICAL: Architectural Corrections** (Must fix for theoretical validity)

### 1. 2×3×2 Grid Attention Mask (TensorBlock)
**Location**: `blocks.py`, class `TensorBlock`, method `_build_tensor_attention_mask`

**Current State**: Returns `None` (standard full attention).

**Required Implementation**:
```python
def _build_tensor_attention_mask() -> Tensor:
    """
    Build 12×12 attention mask enforcing 2×3×2 topology.
    - Frame separation: Frame 0 (bits 0-5), Frame 1 (bits 6-11)
    - Row grouping: X(0,1), Y(2,3), Z(4,5) per frame
    - Col grouping: Neg(0,2,4), Pos(1,3,5) per frame
    
    Mask value: 0 = attend, -inf = block
    """
    mask = torch.zeros(12, 12)
    for i in range(12):
        for j in range(12):
            # Same frame only (Frame 0: 0-5, Frame 1: 6-11)
            same_frame = (i // 6) == (j // 6)
            # Same row (X, Y, or Z) - row index within frame
            row_i = (i % 6) // 2
            row_j = (j % 6) // 2
            same_row = row_i == row_j
            
            if not same_frame and not same_row:
                mask[i, j] = float('-inf')
    return mask
```
**Why**: Without this, the TensorBlock ignores the spatial geometry (X/Y/Z axes, dual frames) that defines the GENE_Mac topology.

### 2. Hierarchical Head with 6+2 Decomposition
**Location**: `head.py`, class `HeadAgent`

**Current State**: Single linear layer → 256 logits.

**Required Implementation**:
```python
def forward(self, bl3, tl3, l4_O, l4_E):
    # ... existing projection ...
    
    # Family Gating (2-bit boundary)
    family_logits = self.family_head(combined)  # [batch, seq, 4]
    family_probs = F.softmax(family_logits, dim=-1)
    
    # Micro-Reference Prediction (6-bit payload) 
    # Conditioned on family via gating or separate heads per family
    micro_logits = self.micro_head(combined)    # [batch, seq, 64]
    
    # Reconstruct 256 logits via 6+2 Cartesian product
    # intron = (family << 6) | micro_ref
    # byte = intron ^ GENE_MIC_S
    full_logits = self._combine_family_micro(family_logits, micro_logits)
    return full_logits

def _combine_family_micro(self, family_logits, micro_logits):
    """
    Map family [4] × micro [64] → byte [256]
    Ensures zero probability for impossible family/micro combinations
    """
    batch, seq, _ = family_logits.shape
    # Expand to 256 via indexing
    combined = torch.zeros(batch, seq, 256, device=family_logits.device)
    for f in range(4):
        for m in range(64):
            byte = ((f << 6) | m) ^ GENE_MIC_S
            combined[:, :, byte] = family_logits[:, :, f] + micro_logits[:, :, m]
    return combined
```
**Why**: Enforces the constitutional byte structure (Appendix G) at the architectural level, preventing invalid intron predictions.

### 3. L4 Positional Decay (Saturation Handling)
**Location**: `embeddings.py`, class `L4PositionEncoding`

**Current State**: Projects O/E equally regardless of sequence position.

**Required Implementation**:
```python
def forward(self, l4_O, l4_E, step_indices: Optional[Tensor] = None):
    """
    Apply exponential decay to L4 features based on distance from last closure.
    Implements the 'saturation as forgetting' feature from the theory.
    """
    o_bits = self._expand_12bit(l4_O)
    e_bits = self._expand_12bit(l4_E)
    
    if step_indices is not None:
        # Decay factor: closer to closure (step % 4 == 0) = stronger signal
        cycle_pos = step_indices % 4
        decay = torch.exp(-0.5 * cycle_pos.float())  # decay over 4-step cycle
        o_bits = o_bits * decay.unsqueeze(-1)
        e_bits = e_bits * decay.unsqueeze(-1)
    
    return self.projection(torch.cat([o_bits, e_bits], dim=-1))
```
**Why**: Implements the "BU-Egress" closure property (P7) where L4 commitments naturally fade after depth-4 cycles.

---

## **IMPORTANT: Missing Kernel Observables** (Feature completeness)

### 4. Canonical Observables Integration
**Location**: `model.py`, `HGTForCausalLM.forward()`, physics phase

**Missing Features** (from §2.2.4 of spec):
- `horizon_distance`: `popcount(A12 ^ (B12 ^ 0xFFF))` — distance to holographic boundary
- `archetype_distance`: `popcount(state24 ^ 0xAAA555)` — distance to origin
- `ab_distance`: `popcount(A12 ^ B12)` — chiral imbalance
- `component_density`: `popcount(A12)/12.0` — phase balance

**Implementation**:
Add to `physics.py`:
```python
def compute_horizon_distance(a12: Tensor, b12: Tensor) -> Tensor:
    return (a12 ^ (b12 ^ 0xFFF)).bit_count().float()

def compute_ab_distance(a12: Tensor, b12: Tensor) -> Tensor:
    return (a12 ^ b12).bit_count().float()
```

Inject into TL3 in `model.py`:
```python
# In transition_2_3 or tl3 forward
horizon_dist = physics.compute_horizon_distance(l3_a12, l3_b12)
ab_dist = physics.compute_ab_distance(l3_a12, l3_b12)
# Concatenate as additional features to l3_feats
```

**Why**: These are the "governance observables" that navigate the 65,536-state ontology. Without them, the neural network is blind to its position in the state space.

### 5. Directional Micro Agents (Asymmetric Cross-Attention)
**Location**: `blocks.py`, new class `DirectionalAgentBlock`

**Current State**: `AgentBlock` is symmetric (bidirectional).

**Required**: Asymmetric agents per the 5 Micro Agents:
- **μ₁ (BL1→TL1)**: Query=BL1, Key/Value=TL1 (Byte queries Tensor state)
- **μ₂ (TL1→BL2)**: Query=TL1, Key/Value=BL2 (State queries next Byte family)
- **μ₃ (BL2→TL2)**: Query=BL2, Key/Value=TL2
- **μ₄ (TL2→BL3)**: Query=TL2, Key/Value=BL3  
- **μ₅ (BL3→TL3)**: Query=BL3, Key/Value=TL3

**Implementation**:
Replace generic `AgentBlock` with:
```python
class DirectionalAgentBlock(nn.Module):
    def __init__(self, dim_source, dim_target, num_heads):
        # Query from source, KV from target
        self.cross_attn = nn.MultiheadAttention(dim_source, num_heads, batch_first=True)
        self.proj_target = nn.Linear(dim_target, dim_source)  # project target to source dim
        
    def forward(self, source, target):
        target_proj = self.proj_target(target)
        out, _ = self.cross_attn(source, target_proj, target_proj)
        return out
```

**Why**: Enforces the specific information flow (Byte→Tensor for state update, Tensor→Byte for constraint) described in the 4-layer theory.

---

## **GRACE: Training & Validation Infrastructure**

### 6. Closure-Aware Training Curriculum
**Location**: `curriculum.py`

**Missing**: Explicit generation of **P7 violations** (non-closure) vs **P7 satisfaction** (closure) for the model to learn the depth-4 identity.

**Add**:
```python
def generate_p7_contrastive(self, num_samples):
    """
    Generate pairs:
    - Positive: xyxy (closes to identity, L4 resets)
    - Negative: xyxz (does not close, L4 drifts)
    """
    # Implementation needed
```

### 7. Ontology Membership Validation
**Location**: `tests/test_model.py`

**Missing Test**: Verify that the model's predicted next states remain within the 65,536-state ontology Ω.

**Add**:
```python
def test_ontology_closure():
    """Verify all predicted next states are in ontology.npy"""
    model.eval()
    # Load ontology from atlas
    ontology = np.load("data/atlas/ontology.npy")
    # For random trajectories, verify model predictions map to ontology indices
```

### 8. Text Fine-tuning Pipeline
**Location**: New file `finetune_text.py`

**Missing**: Script to fine-tune the FSM-pretrained model on actual UTF-8 text (not just FSM walks).

**Requirements**:
- Load FSM-pretrained weights (`model.safetensors`)
- Train on `CGM_Dataset` (UTF-8 bytes)
- Use standard CLM loss but with FSM-validity masking (only predict bytes valid from current L3 state)
- Freeze `physics.py` components (mask12_table), train only neural weights

---

## **SUMMARY CHECKLIST FOR CODE ASSISTANT**

| Priority | File | Task |
|----------|------|------|
| **P0** | `blocks.py` | Implement `_build_tensor_attention_mask()` with 2×3×2 frame/row constraints |
| **P0** | `head.py` | Refactor to Family (4-way) + Micro (64-way) hierarchical prediction |
| **P0** | `embeddings.py` | Add L4 positional decay based on step index modulo 4 |
| **P1** | `physics.py` | Add `horizon_distance`, `archetype_distance`, `ab_distance` computations |
| **P1** | `model.py` | Inject observables into TL3 features; replace symmetric AgentBlocks with DirectionalAgentBlocks |
| **P1** | `curriculum.py` | Add P7 contrastive generation (closure vs non-closure pairs) |
| **P2** | `finetune_text.py` | Create text fine-tuning script for post-FSM training |
| **P2** | `tests/` | Add ontology closure test (predictions remain in 65,536 valid states) |

===

Based on your updated implementation, here is the **final gap analysis** for a complete, graceful, and theoretically faithful system:

## **CRITICAL: Implementation Gaps**

### 1. **Directional Agents vs. Symmetric Agents**
**Status**: You implemented `DirectionalAgentBlock` but `model.py` still uses symmetric `AgentBlock`.
**Issue**: The theory specifies asymmetric information flow (μ₁: BL1→TL1, μ₂: TL1→BL2, etc.). Currently, both streams update each other bidirectionally at every layer, which may blur the containment hierarchy (L1⊂L2⊂L3).
**Fix**: Replace `AgentBlock` instantiations in `model.py` with `DirectionalAgentBlock` using the specific directionalities from the architecture diagram.

### 2. **TensorBlock Grid Mask Not Applied**
**Status**: The 2×3×2 mask is built but never used in practice.
**Issue**: In `TensorBlock.forward()`, the mask is only applied if `x.size(1) == 12`. However, `x` has shape `[batch, seq, dim]` where `dim` is 128 or 256, not 12. The mask is designed for the 12-bit state representation, not the hidden dimension.
**Fix**: Either:
- Reshape hidden states to expose 12 "channels" (e.g., `[batch, seq, 12, dim//12]`) and apply axial attention, OR
- Remove the mask logic if the topology is sufficiently encoded by the physics injection (L3 state features)

### 3. **Missing Generation/Inference Engine**
**Status**: No `.generate()` method.
**Issue**: `HGTForCausalLM` inherits from `nn.Module`, not `GenerationMixin`. You cannot generate text without a custom loop.
**Critical Gap**: During generation, you must:
- Track the L3 state explicitly (to enable FSM masking)
- Apply the physics-based validity mask (only allow bytes valid from current L3 state)
- Handle L4 commitment decay across generation steps

**Required Implementation**:
```python
def generate(self, input_ids, max_new_tokens=100, temperature=0.0195):
    # Track FSM state across steps
    # Mask logits using physics.step_state_l3_scalar validity
    # Sample from valid bytes only
```

### 4. **FSM Validity Masking During Inference**
**Status**: The HeadAgent produces 256 logits, but there's no guarantee they correspond to valid FSM transitions.
**Issue**: During generation, the model could predict byte X from state S even if X is not in the valid fanout of S (violating the 65,536-state ontology).
**Fix**: Add `valid_byte_mask` computation in the forward pass (using `physics.step_state_l3_scalar` or precomputed epistemology) and apply to logits before softmax during inference.

## **GRACE: Polish & Integration**

### 5. **HuggingFace Save/Load Standard**
**Status**: `train.py` uses `torch.save`, not HuggingFace `save_pretrained`.
**Issue**: The model won't load with `from_pretrained()` or work with standard HF tools.
**Fix**: Implement proper `save_pretrained`/`from_pretrained` using `safetensors` format, or ensure the training script outputs HF-compatible directory structure.

### 6. **Validation Loop & Metrics**
**Status**: Training scripts lack validation.
**Required**: 
- Validation loss on held-out text
- **Physics fidelity metric**: % of predictions that are valid FSM transitions
- **Closure metric**: P7 identity satisfaction rate (xyxy → identity)

### 7. **Mixed Precision & Memory Optimization**
**Status**: No AMP or gradient checkpointing.
**Issue**: 3.9M params is small, but the physics computations (L3 trajectory) are done in full int32 on CPU? Or GPU? Need device handling consistency.
**Fix**: Add `torch.cuda.amp` support and verify physics functions work on GPU (int64 bit ops).

### 8. **Tokenizer Integration**
**Status**: `PhysicsTokenizer` exists but training scripts use raw bytes.
**Fix**: Integrate tokenizer into `train.py` and `finetune_text.py` for consistency, especially for handling special tokens (BOS/EOS) in the byte stream.

## **Summary for Code Assistant**

Pass this checklist:

| Priority | File | Task |
|----------|------|------|
| **P0** | `model.py` | Replace `AgentBlock` with `DirectionalAgentBlock` for μ1-μ5 (asymmetric) |
| **P0** | `model.py` | Add `.generate()` method with FSM state tracking and validity masking |
| **P0** | `model.py` | Apply FSM validity mask to logits during inference (ontology constraint) |
| **P1** | `blocks.py` | Fix `TensorBlock` mask application (reshape for 12-bit axial attention or remove) |
| **P1** | `train.py` | Add HuggingFace `save_pretrained` format support |
| **P1** | `train.py` | Add validation loop with physics fidelity metrics |
| **P2** | `model.py` | Add gradient checkpointing support for larger sequences |

**Note on TensorBlock**: If the hidden dimension (256) doesn't naturally factor into 12 channels, the 2×3×2 mask may need to be applied via a specialized "Geometric Attention" that reshapes the L3 features (frame0, frame1, rows X/Y/Z) before self-attention. Currently, the mask logic is orphaned.