"""
GyroSI Kernel v0.9.12.0 - Complete Physics Implementation for Language Modeling

This implements the Common Governance Model (CGM) through a physics-first approach
to language processing. Replaces transformer matrix multiplications with pure state
transitions through a finite 48-bit manifold.

Key Features:
- Downloads and uses any HuggingFace language model
- Converts model weights to compressed state sequences via Fold
- All generation via pure physical resonance (no scoring, no heuristics)
- Sparse holographic storage (only deviations from baseline)
- Complete CS asymmetric emission with canonical broadcast masks
- Forward-only cycle gating with correct defect calculation
- Proper SmolLM tokenizer integration
- Virtual tokens from model weights used in generation
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open


# ============================================================================
# CORE CONSTANTS
# ============================================================================

GENE_Mic_S = 0xAA  # Holographic topology constant

GENE_Mac_S = np.array(
    [
        # Layer 0: 0° phase
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        # Layer 1: 180° phase
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        # Layer 2: 360° phase
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        # Layer 3: 540° phase
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ],
    dtype=np.int8,
)

# Bit family masks per CGM
EXON_LI_MASK = 0b01000010  # Bits 1, 6 - UNA (Parity/Reflection)
EXON_FG_MASK = 0b00100100  # Bits 2, 5 - ONA (Forward Gyration)
EXON_BG_MASK = 0b00011000  # Bits 3, 4 - BU (Backward Gyration)
EXON_L0_MASK = 0b10000001  # Bits 0, 7 - Anchors (Boundaries)
EXON_DYNAMIC_MASK = EXON_LI_MASK | EXON_FG_MASK | EXON_BG_MASK

# CGM Stage angles in radians
STAGE_ANGLES = {
    "CS": np.pi / 2,  # Common Source
    "UNA": np.pi / 4,  # Unity Non-Absolute
    "ONA": np.pi / 4,  # Opposition Non-Absolute
    "BU_IN": 0.0,  # Balance Universal - Ingress
    "BU_EG": np.pi / 2,  # Balance Universal - Egress
}

# Theta thresholds for stage detection
THETA_CS = 0.1  # Near zero for CS
THETA_UNA = 0.785  # π/4 for UNA
THETA_ONA = 1.0  # Between UNA and BU
THETA_BU_IN = 1.3  # BU Ingress
THETA_BU_EG = 1.5  # BU Egress

STAGE_ORDER = ["CS", "UNA", "ONA", "BU_IN", "BU_EG"]

# Canonical bit ordering for tensor packing
TENSOR_BIT_ORDER = []
for layer in range(4):
    for frame in range(2):
        for row in range(3):
            for col in range(2):
                TENSOR_BIT_ORDER.append((layer, frame, row, col))


# ============================================================================
# CORE PHYSICS FUNCTIONS
# ============================================================================


def fold(a: int, b: int) -> int:
    """
    The Monodromic Fold (⋄), the path-dependent learning operator.

    Canonical Form: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
    Algebraic Normal Form: a ⋄ b = ¬a ∧ b

    These are mathematically identical through Boolean algebra.
    Non-associative, path-dependent learning operator.
    This is the ONLY learning/integration operator in the system.
    """
    a &= 0xFF
    b &= 0xFF
    negated_b = (~b) & 0xFF
    gyration = b ^ (a & negated_b)
    return (a ^ gyration) & 0xFF


def fold_sequence(values: List[int], start_state: int = 0) -> int:
    """Apply Monodromic Fold sequentially (path-dependent)."""
    result = start_state & 0xFF
    for value in values:
        result = fold(result, value & 0xFF)
    return result


def transcribe_byte(byte: int) -> int:
    """ψ isomorphism: byte → intron via XOR 0xAA."""
    return (byte ^ GENE_Mic_S) & 0xFF


def untranscribe_byte(intron: int) -> int:
    """ψ⁻¹ isomorphism: intron → byte via XOR 0xAA."""
    return (intron ^ GENE_Mic_S) & 0xFF


def tensor_to_int(tensor: np.ndarray) -> int:
    """Convert 48-element tensor to packed integer using canonical bit order."""
    if tensor.shape != (4, 2, 3, 2):
        tensor = tensor.reshape(4, 2, 3, 2)

    result = 0
    for bit_pos, (layer, frame, row, col) in enumerate(TENSOR_BIT_ORDER):
        if tensor[layer, frame, row, col] == -1:
            result |= 1 << bit_pos

    return result


def int_to_tensor(state_int: int) -> np.ndarray:
    """Convert packed integer to 48-element tensor using canonical bit order."""
    if state_int >= (1 << 48) or state_int < 0:
        raise ValueError(f"state_int {state_int} out of bounds for 48-bit")

    tensor = np.ones((4, 2, 3, 2), dtype=np.int8)

    for bit_pos, (layer, frame, row, col) in enumerate(TENSOR_BIT_ORDER):
        if (state_int >> bit_pos) & 1:
            tensor[layer, frame, row, col] = -1

    return tensor


def compute_exon_from_state(state_int: int) -> int:
    """Compute 8-bit exon from 48-bit state via helical folding."""
    # Split into 6 bytes and fold
    b = [(state_int >> (i * 8)) & 0xFF for i in range(6)]

    # Fold opposites, then fold results
    p1 = fold(b[0], b[3])
    p2 = fold(b[1], b[4])
    p3 = fold(b[2], b[5])

    # Final fold
    exon = fold(fold(p1, p2), p3)

    # Physics-based fallback for zero
    if exon == 0:
        exon = fold(GENE_Mic_S, 0x01)

    return exon


def generate_intron_broadcast_masks() -> np.ndarray:
    """Generate 256x48 broadcast masks by repeating intron across 6 bytes."""
    masks = np.zeros((256, 48), dtype=np.uint8)

    for intron in range(256):
        for j in range(6):
            byte_val = intron & 0xFF
            start_bit = j * 8
            for bit in range(8):
                if byte_val & (1 << bit):
                    masks[intron, start_bit + bit] = 1

    return masks


# ============================================================================
# MODEL DOWNLOAD AND WEIGHT CONVERSION
# ============================================================================


def download_model(model_name: str = None, force_reconvert: bool = False, debug: bool = False):
    """Download and convert any HuggingFace language model to physics format."""
    if model_name is None:
        model_name = "HuggingFaceTB/SmolLM-360M"

    # Create kernel directory structure
    kernel_dir = Path(__file__).parents[1] / "memories" / "kernel"
    kernel_dir.mkdir(exist_ok=True)

    # Create model-specific subdirectory
    model_key = model_name.replace("/", "_").replace("-", "_")
    model_dir = kernel_dir / model_key
    model_dir.mkdir(exist_ok=True)

    # Define cache files
    tokenizer_cache = model_dir / "tokenizer.json"
    config_cache = model_dir / "config.json"
    conversion_cache = model_dir / "conversion_meta.json"

    try:
        # Check if we have cached conversion metadata
        if not force_reconvert and conversion_cache.exists():
            if debug:
                print(f"📁 Found cached conversion for {model_name}")

            # Check if all required files exist
            if all(f.exists() for f in [tokenizer_cache, config_cache]):
                if debug:
                    print(f"  → Loading cached tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

                if debug:
                    print(f"  → Loading cached weights...")
                # Load cached weights from chunks
                chunk_meta_file = model_dir / "chunk_meta.json"
                weights = {}
                if chunk_meta_file.exists():
                    with open(chunk_meta_file, "r") as f:
                        chunk_meta = json.load(f)

                    for chunk_idx in range(chunk_meta["num_chunks"]):
                        chunk_file = model_dir / f"weights_chunk_{chunk_idx}.npz"
                        if chunk_file.exists():
                            chunk_data = np.load(str(chunk_file), allow_pickle=True)
                            for key in chunk_data.files:
                                weights[key] = chunk_data[key]
                            if debug:
                                print(f"    Loaded chunk {chunk_idx + 1}/{chunk_meta['num_chunks']}")

                if debug:
                    print(f"  → Loading cached config...")
                with open(config_cache, "r") as f:
                    model_config = json.load(f)

                if debug:
                    print(f"✅ Using cached conversion: {len(weights)} weight tensors")
                return {
                    "tokenizer": tokenizer,
                    "weights": weights,
                    "vocab_size": tokenizer.vocab_size,
                    "model_config": model_config,
                    "resolved_model_name": model_name,
                    "model_key": model_key,
                    "cached": True,
                }

        # Download from HuggingFace if not cached
        if debug:
            print(f"📥 Downloading {model_name} model files...")

        # Download tokenizer
        if debug:
            print(f"  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save tokenizer to cache
        tokenizer.save_pretrained(str(model_dir))
        if debug:
            print(f"  → Saved tokenizer to cache")

        # Download raw model files
        if debug:
            print(f"  → Downloading model files...")
        model_files = snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.safetensors", "*.bin", "config.json"],
            ignore_patterns=["*.md", "*.txt", "*.git*"],
        )

        # Load config
        config_path = f"{model_files}/config.json"
        with open(config_path, "r") as f:
            model_config = json.load(f)

        # Save config to cache
        with open(config_cache, "w") as f:
            json.dump(model_config, f, indent=2)
        if debug:
            print(f"  → Saved config to cache")

        # Collect weight file paths
        safetensor_files = []
        pytorch_model_path = None
        for filename in os.listdir(model_files):
            if filename.endswith(".safetensors"):
                safetensor_files.append(os.path.join(model_files, filename))
            elif filename == "pytorch_model.bin":
                pytorch_model_path = os.path.join(model_files, filename)
        safetensor_files.sort()

        # Convert weights and cache them
        weights = {}
        tensors_processed = 0

        if debug:
            print(f"  → Converting weights...")
        for st_path in safetensor_files:
            try:
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        if isinstance(tensor, torch.Tensor):
                            try:
                                arr = tensor.detach().cpu().numpy()
                            except TypeError:
                                arr = tensor.detach().cpu().float().numpy()

                            weights[key] = arr
                            tensors_processed += 1
                            if debug and tensors_processed % 50 == 0:
                                print(f"    Progress: {tensors_processed} tensors processed")
            except Exception as e:
                if debug:
                    print(f"    ⚠️ Failed to read {st_path}: {e}")

        # Save converted weights to cache in chunks
        if debug:
            print(f"  → Saving {len(weights)} weight tensors to cache...")
        chunk_size = 50
        weight_keys = list(weights.keys())
        for i in range(0, len(weight_keys), chunk_size):
            chunk_keys = weight_keys[i : i + chunk_size]
            chunk_weights = {k: weights[k] for k in chunk_keys}
            chunk_file = model_dir / f"weights_chunk_{i//chunk_size}.npz"
            np.savez_compressed(str(chunk_file), **chunk_weights)
            if debug:
                print(f"    Saved chunk {i//chunk_size + 1}/" f"{(len(weight_keys) + chunk_size - 1)//chunk_size}")

        # Save chunk metadata
        chunk_meta = {
            "num_chunks": (len(weight_keys) + chunk_size - 1) // chunk_size,
            "chunk_size": chunk_size,
            "total_tensors": len(weight_keys),
        }
        with open(model_dir / "chunk_meta.json", "w") as f:
            json.dump(chunk_meta, f)

        # Save conversion metadata
        conversion_meta = {
            "model_name": model_name,
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_tensors": tensors_processed,
            "tensor_names": list(weights.keys()),
        }
        with open(conversion_cache, "w") as f:
            json.dump(conversion_meta, f, indent=2)

        if debug:
            print(f"✅ Downloaded and converted {model_name}: {tensors_processed} weight tensors")

        return {
            "tokenizer": tokenizer,
            "weights": weights,
            "vocab_size": tokenizer.vocab_size,
            "model_config": model_config,
            "resolved_model_name": model_name,
            "model_key": model_key,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}: {e}")


def quantize_and_compress_weights(weights: np.ndarray, max_size: int = 10000) -> Tuple[List[int], float]:
    """Convert weights to compressed intron sequence using pure physics."""
    # Flatten and limit size
    flat = weights.flatten()
    if len(flat) > max_size:
        flat = flat[:max_size]

    # Quantize to int8
    scale = np.max(np.abs(flat)) / 127.0 if np.max(np.abs(flat)) > 0 else 1.0
    quantized = np.clip(flat / scale, -128, 127).astype(np.int8)

    # Convert to unsigned for delta encoding
    unsigned = quantized.view(np.uint8)

    # Delta encode for compression
    deltas = np.zeros(len(unsigned), dtype=np.uint8)
    prev = 0
    for i in range(len(unsigned)):
        deltas[i] = (int(unsigned[i]) - prev) & 0xFF
        prev = int(unsigned[i])

    # Convert to introns
    introns = [transcribe_byte(d) for d in deltas]

    # Compress via Fold sequence
    compressed = []
    for i in range(0, len(introns), 8):
        chunk = introns[i : i + 8]
        compressed.append(fold_sequence(chunk, start_state=GENE_Mic_S))

    return compressed, scale


# ============================================================================
# GYRO KERNEL - COMPLETE PHYSICS IMPLEMENTATION
# ============================================================================


class GyroKernel:
    """GyroSI Kernel v0.9.12.0 - Complete Physics Implementation

    Implements the Common Governance Model (CGM) as a functional language model
    that replaces transformer architecture with gyroscopic intelligence.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        debug: bool = False,
        model_name: str = None,
        force_reconvert: bool = False,
    ):
        """Initialize kernel with physics tables and model."""
        if base_path is None:
            base_path = Path(__file__).parents[1] / "memories"

        self.base_path = base_path
        self.debug = debug
        self.model_name = model_name or "HuggingFaceTB/SmolLM-360M"
        self.force_reconvert = force_reconvert

        # Load physics tables
        self._load_physics_tables()

        # Load broadcast masks
        self._load_broadcast_masks()

        # Find CS state (minimum theta)
        self._find_cs_state()

        # Initialize archetypal state
        self.archetypal_state_int = tensor_to_int(GENE_Mac_S)

        # Current state
        self.current_state_index = self.CS_STATE_INDEX
        self.stage_index = 0

        # Memory: orbit_rep -> token_id -> mask
        self.memory: Dict[int, Dict[int, int]] = {}

        # Path memory
        self.path_memory = GENE_Mic_S

        # Model knowledge (compressed weights)
        self.virtual_tokens: Dict[Tuple[str, int], int] = {}

        # Token → state mappings
        self.token_state_map: Dict[int, int] = {}
        self.token_post_state_index: Dict[int, int] = {}
        self.token_exon_cache: Dict[int, int] = {}

        # Precomputed resonance table
        self._resonance_table = np.zeros((256, 256), dtype=np.uint8)
        for a in range(256):
            for b in range(256):
                self._resonance_table[a, b] = fold(a, b)

        # Orbit candidates
        self._orbit_candidates: Dict[int, List[int]] = {}

        # UNA pool for CS emission
        self._precompute_una_pool()

        # Load model and tokenizer
        model_data = download_model(self.model_name, self.force_reconvert, debug=self.debug)
        if not model_data or not model_data["tokenizer"]:
            raise RuntimeError("Failed to load tokenizer - cannot operate without it")

        self.model_name = model_data.get("resolved_model_name", self.model_name)
        self.model_key = model_data.get("model_key", self.model_name.replace("/", "_").replace("-", "_"))
        self.tokenizer = model_data["tokenizer"]

        # Set special tokens from tokenizer
        self._set_special_tokens()

        # Build valid tokens
        self.valid_tokens = set(range(self.tokenizer.vocab_size))

        # Import model weights
        self._import_model_weights(model_data)

        # Build token post-states and exons
        self._build_or_load_token_post_states()

        # Build orbit candidates
        self._build_orbit_candidates()

        # Add virtual tokens to orbit candidates
        self._integrate_virtual_tokens()

        # Statistics
        self.stats = {"tokens_learned": 0, "memory_entries": 0, "orbits_discovered": 0, "generation_steps": 0}

        if self.debug:
            print(f"\n🧬 GyroSI Kernel v0.9.12.0 initialized")
            print(f"📍 CS state: index={self.CS_STATE_INDEX}, " f"θ={float(self.theta[self.CS_STATE_INDEX]):.4f}")
            print(f"📊 Ontology: {len(self.ontology):,} states")
            print(f"💫 Virtual tokens: {len(self.virtual_tokens):,}")
            print(f"🤖 Model: {self.model_name}")
            print(f"📝 Tokenizer vocab size: {self.tokenizer.vocab_size}")

    def _load_physics_tables(self) -> None:
        """Load all physics tables with memory mapping."""
        meta_path = self.base_path / "public" / "meta"

        # Ontology (state integers)
        ontology_path = meta_path / "ontology_keys.npy"
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")
        self.ontology = np.load(ontology_path, mmap_mode="r")

        # Epistemology (state transitions)
        epistemology_path = meta_path / "epistemology.npy"
        if not epistemology_path.exists():
            raise FileNotFoundError(f"Epistemology not found: {epistemology_path}")
        self.epistemology = np.load(epistemology_path, mmap_mode="r")

        # Theta (angular divergence)
        theta_path = meta_path / "theta.npy"
        if not theta_path.exists():
            raise FileNotFoundError(f"Theta not found: {theta_path}")
        self.theta = np.load(theta_path, mmap_mode="r")

        # Phenomenology (orbit mapping)
        pheno_path = meta_path / "phenomenology_map.npy"
        if not pheno_path.exists():
            raise FileNotFoundError(f"Phenomenology not found: {pheno_path}")
        self.phenomenology = np.load(pheno_path, mmap_mode="r")

        # Orbit sizes
        orbit_sizes_path = meta_path / "orbit_sizes.npy"
        if not orbit_sizes_path.exists():
            raise FileNotFoundError(f"Orbit sizes not found: {orbit_sizes_path}")
        self.orbit_sizes = np.load(orbit_sizes_path, mmap_mode="r")

    def _load_broadcast_masks(self) -> None:
        """Load broadcast masks for boundary layer operations with CS as extra-phenomenal axiom."""
        meta_path = self.base_path / "public" / "meta"
        broadcast_masks_path = meta_path / "intron_broadcast_masks.npy"

        if broadcast_masks_path.exists():
            self.INTRON_BROADCAST_MASKS = np.load(broadcast_masks_path, mmap_mode="r")
            if self.debug:
                print(f"✅ Loaded {self.INTRON_BROADCAST_MASKS.shape[0]} broadcast masks")
        else:
            # Generate if not found
            self.INTRON_BROADCAST_MASKS = generate_intron_broadcast_masks()
            if self.debug:
                print(f"✅ Generated {self.INTRON_BROADCAST_MASKS.shape[0]} broadcast masks")

    def _find_cs_state(self) -> None:
        """Find the CS state (minimum theta) as extra-phenomenal reference point."""
        min_theta_idx = int(np.argmin(self.theta))
        min_theta = float(self.theta[min_theta_idx])

        self.CS_STATE_INDEX = min_theta_idx
        self.CS_STATE_INT = int(self.ontology[min_theta_idx])

        if self.debug:
            print(f"✅ Found CS state: index={min_theta_idx}, θ={min_theta:.4f}")

    def _precompute_una_pool(self) -> None:
        """Precompute UNA states for boundary layer operations with CS axiom."""
        # Find states with theta close to π/4
        target_theta = np.pi / 4
        tight_tolerance = 0.05  # Tighter tolerance for sharper CS emission
        fallback_tolerance = 0.1

        # Try tight tolerance first
        self._UNA_pool = np.argwhere(np.abs(self.theta - target_theta) < tight_tolerance).astype(np.int32).ravel()

        # If too small, fallback to looser tolerance
        if len(self._UNA_pool) < 10:  # Minimum viable pool size
            self._UNA_pool = (
                np.argwhere(np.abs(self.theta - target_theta) < fallback_tolerance).astype(np.int32).ravel()
            )

        # Final fallback: use states with theta in UNA range
        if len(self._UNA_pool) == 0:
            self._UNA_pool = np.argwhere((self.theta > THETA_CS) & (self.theta < THETA_ONA)).astype(np.int32).ravel()

        if self.debug:
            print(f"✅ Precomputed UNA pool: {len(self._UNA_pool)} candidates")

    def _set_special_tokens(self) -> None:
        """Set special tokens from SmolLM tokenizer."""
        # SmolLM uses <|endoftext|> (id 0) for both BOS and EOS
        self.CLS_TOKEN = self.tokenizer.convert_tokens_to_ids("<|im_start|>")  # 1
        self.SEP_TOKEN = self.tokenizer.convert_tokens_to_ids("<|im_end|>")  # 2
        self.PAD_TOKEN = self.tokenizer.pad_token_id or 0

        # Chat markers
        self.IM_START = 1  # <|im_start|>
        self.IM_END = 2  # <|im_end|>

        if self.debug:
            print(
                f"📌 Special tokens: CLS={self.CLS_TOKEN}, SEP={self.SEP_TOKEN}, "
                f"IM_START={self.IM_START}, IM_END={self.IM_END}"
            )

    def _import_model_weights(self, model_data: dict) -> None:
        """Import model weights and convert to virtual tokens."""
        weights = model_data.get("weights", {})

        if model_data.get("cached", False):
            # Load cached virtual tokens
            vt_cache = self.base_path / "kernel" / self.model_key / "virtual_tokens.npz"
            if vt_cache.exists():
                try:
                    vt_data = np.load(str(vt_cache), allow_pickle=True)
                    self.virtual_tokens = vt_data.get("virtual_tokens", {}).item()
                    if self.debug:
                        print(f"✅ Loaded {len(self.virtual_tokens):,} cached virtual tokens")
                except Exception:
                    pass

        # If not loaded from cache, compress weights
        if not self.virtual_tokens and weights:
            if self.debug:
                print(f"📦 Converting model weights to virtual tokens...")

            for key, arr in weights.items():
                compressed_bytes, _ = quantize_and_compress_weights(arr)
                for pos, byte_val in enumerate(compressed_bytes):
                    self.virtual_tokens[(key, pos)] = byte_val

            # Cache virtual tokens
            try:
                vt_cache = self.base_path / "kernel" / self.model_key / "virtual_tokens.npz"
                vt_cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(vt_cache), virtual_tokens=self.virtual_tokens)
                if self.debug:
                    print(f"💾 Cached {len(self.virtual_tokens):,} virtual tokens")
            except Exception:
                pass

        # Find embedding matrix for projection
        embeddings_buffer = None
        for key, arr in weights.items():
            if arr.ndim == 2 and "embed" in key.lower() and "weight" in key.lower():
                embeddings_buffer = arr
                break

        # Project embeddings to token states
        if embeddings_buffer is not None:
            self._project_embeddings_to_token_states(embeddings_buffer)

    def _project_embeddings_to_token_states(self, embeddings: np.ndarray) -> None:
        """Project token embeddings to 48-bit states via physics fold."""
        # Check cache first
        cache_file = self.base_path / "kernel" / self.model_key / "embedding_projection.npz"

        if cache_file.exists() and not self.force_reconvert:
            try:
                cached_data = np.load(str(cache_file), allow_pickle=True)
                self.token_state_map = cached_data["token_state_map"].item()
                if self.debug:
                    print(f"✅ Loaded cached embedding projection: " f"{len(self.token_state_map)} tokens")
                return
            except Exception:
                pass

        if self.debug:
            print(f"  → Computing embedding projection...")

        vocab = min(self.tokenizer.vocab_size, embeddings.shape[0])

        for token_id in range(vocab):
            try:
                vec = embeddings[token_id]
                max_abs = float(np.max(np.abs(vec))) if np.max(np.abs(vec)) > 0 else 1.0
                q = np.clip((vec / max_abs) * 127.0, -128, 127).astype(np.int8).view(np.uint8)

                # Produce 6 bytes via fold
                six_bytes = []
                for k in range(6):
                    acc = GENE_Mic_S
                    for idx in range(k, q.shape[0], 6):
                        acc = fold(acc, transcribe_byte(int(q[idx])))
                    six_bytes.append(acc & 0xFF)

                # Pack into 48-bit state
                state_int = 0
                for byte_index, byte_val in enumerate(six_bytes):
                    state_int |= (int(byte_val) & 0xFF) << (8 * byte_index)
                state_int &= (1 << 48) - 1

                self.token_state_map[token_id] = state_int
            except Exception:
                continue

        if self.debug:
            print(f"✅ Token states built: {len(self.token_state_map):,}")

        # Cache the projection
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(cache_file), token_state_map=self.token_state_map)
            if self.debug:
                print(f"💾 Cached embedding projection")
        except Exception:
            pass

    def _build_or_load_token_post_states(self) -> None:
        """Build or load token post-states and exons."""
        # Check cache
        cache_dir = self.base_path / "kernel" / self.model_key
        post_path = cache_dir / "token_post_states.npy"
        exons_path = cache_dir / "token_exons.npy"

        if post_path.exists() and exons_path.exists():
            # Load from cache
            post_idx = np.load(str(post_path), mmap_mode="r")
            exons = np.load(str(exons_path), mmap_mode="r")

            # Rehydrate dictionaries
            self.token_post_state_index = {int(i): int(s) for i, s in enumerate(post_idx) if s >= 0}
            self.token_exon_cache = {int(i): int(e) for i, e in enumerate(exons) if e >= 0}

            if self.debug:
                print(f"✅ Loaded token post-states/exons from cache")
        else:
            # Build fresh
            self._build_token_post_states()

            # Cache them
            vocab = self.tokenizer.vocab_size
            post_idx = np.full((vocab,), -1, dtype=np.int64)
            exons = np.full((vocab,), -1, dtype=np.int16)

            for tok, st in self.token_post_state_index.items():
                post_idx[int(tok)] = int(st)
            for tok, ex in self.token_exon_cache.items():
                exons[int(tok)] = int(ex)

            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(post_path), post_idx)
            np.save(str(exons_path), exons)

            if self.debug:
                print(f"💾 Cached token post-states/exons")

        # Add special tokens
        for tok in [self.CLS_TOKEN, self.SEP_TOKEN, self.PAD_TOKEN]:
            if tok not in self.token_exon_cache:
                s = self.token_post_state_index[tok]
                self.token_exon_cache[tok] = compute_exon_from_state(int(self.ontology[s]))

    def _build_token_post_states(self) -> None:
        """Build token post-states using CS as extra-phenomenal reference."""
        if self.debug:
            print(f"  → Building token post-states...")

        self.token_post_state_index.clear()
        self.token_exon_cache.clear()

        cs_index = self.CS_STATE_INDEX

        for token_id in range(self.tokenizer.vocab_size):
            if token_id not in self.valid_tokens:
                continue

            # Convert token to introns
            leb = self._id_to_uleb128(token_id)
            introns = [transcribe_byte(b) for b in leb]

            # Walk state
            s = cs_index
            for intron in introns:
                s = int(self.epistemology[s, intron & 0xFF])

            self.token_post_state_index[token_id] = s

            # Compute exon
            state_int = int(self.ontology[s])
            exon = compute_exon_from_state(state_int)
            self.token_exon_cache[token_id] = exon

        if self.debug:
            print(f"✅ Token post-states built: {len(self.token_post_state_index):,}")

    def _build_orbit_candidates(self) -> None:
        """Build orbit-based candidate lists."""
        self._orbit_candidates.clear()

        for token_id, post_idx in self.token_post_state_index.items():
            orbit_rep = int(self.phenomenology[post_idx])

            if orbit_rep not in self._orbit_candidates:
                self._orbit_candidates[orbit_rep] = []

            self._orbit_candidates[orbit_rep].append(token_id)

        if self.debug:
            print(f"✅ Built orbit candidates: {len(self._orbit_candidates)} orbits")

    def _integrate_virtual_tokens(self) -> None:
        """Add virtual tokens to orbit candidates."""
        if not self.virtual_tokens:
            return

        # Add virtual tokens as synthetic candidates
        for (key, pos), byte_val in self.virtual_tokens.items():
            # Create Traceable virtual token ID
            vt_id = self.tokenizer.vocab_size + hash((key, pos))

            # Find the state derived from this virtual token's byte_val (intron) applied from CS
            derived_state_index = int(self.epistemology[self.CS_STATE_INDEX, byte_val])
            orbit = int(self.phenomenology[derived_state_index])

            if orbit not in self._orbit_candidates:
                self._orbit_candidates[orbit] = []

            # Limit virtual tokens per orbit
            if len(self._orbit_candidates[orbit]) < 5000:
                self._orbit_candidates[orbit].append(vt_id)
                self.token_exon_cache[vt_id] = byte_val

        if self.debug:
            print(f"✅ Integrated virtual tokens into candidates")

    def _id_to_uleb128(self, x: int) -> List[int]:
        """Encode integer to unsigned LEB128 bytes."""
        out = []
        val = int(x)
        while True:
            b = val & 0x7F
            val >>= 7
            if val:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
        return out

    def token_to_introns(self, token_id: int) -> List[int]:
        """Convert token ID to introns using LEB128 encoding."""
        if token_id >= self.tokenizer.vocab_size:
            return []

        leb = self._id_to_uleb128(token_id)
        return [transcribe_byte(b) for b in leb]

    def _get_stage(self, state_index: int) -> str:
        """Get CGM stage from theta value."""
        theta = float(self.theta[state_index])

        if theta < THETA_CS:
            return "CS"
        elif theta < THETA_UNA:
            return "UNA"
        elif theta < THETA_ONA:
            return "ONA"
        elif theta < THETA_BU_IN:
            return "BU_IN"
        else:
            return "BU_EG"

    def _update_stage(self, state_index: int) -> None:
        """Update stage based on state's theta."""
        theta = float(self.theta[state_index])

        # Use _get_stage to determine the current stage and set stage_index
        stage = self._get_stage(state_index)
        self.stage_index = STAGE_ORDER.index(stage)

    def _apply_intron_and_gate(self, state_index: int, intron: int) -> int:
        """Apply intron using generic physics with cycle gating."""
        # Generic transition for all states
        if 0 <= state_index < len(self.epistemology):
            next_index = int(self.epistemology[state_index, intron & 0xFF])
            current_stage_idx = STAGE_ORDER.index(self._get_stage(state_index))
            next_stage_idx = STAGE_ORDER.index(self._get_stage(next_index))

            # Block only if stage regresses
            if next_stage_idx < current_stage_idx:
                if self.debug:
                    print(
                        f"  ↪️ Cycle gating blocked regression: {STAGE_ORDER[current_stage_idx]} → {STAGE_ORDER[next_stage_idx]}"
                    )
                return state_index

            # Update path memory
            self.path_memory = fold(self.path_memory, intron)

            return next_index

        return state_index

    def _get_candidates_for_state(self, state_index: int) -> List[int]:
        """Get candidate tokens for current state (pure orbit lookup)."""
        orbit = int(self.phenomenology[state_index])
        candidates = self._orbit_candidates.get(orbit, [])

        if not candidates:
            raise RuntimeError(f"No candidates found for orbit {orbit} at state {state_index}.")

        # Filter to valid range
        candidates = [c for c in candidates if c < self.tokenizer.vocab_size * 2]

        return candidates

    def _resonance_defect(self, token_id: int) -> int:
        """Calculate pure Fold resonance defect."""
        # Use weight_lens if available
        if hasattr(self, "weight_lens") and self.weight_lens is not None:
            try:
                exon = self.weight_lens.exon_for_token(int(token_id))
                defect = self._resonance_table[self.path_memory & 0xFF, exon & 0xFF]
                return defect
            except Exception:
                pass

        # Fallback to cached exon or default
        exon = self.token_exon_cache.get(token_id, GENE_Mic_S)
        defect = self._resonance_table[self.path_memory & 0xFF, exon & 0xFF]
        return defect

    def learn_token(self, token_id: int) -> None:
        """Learn a token using sparse BU hinge physics."""
        if token_id not in self.valid_tokens:
            return

        # Track statistics
        self.stats["tokens_learned"] += 1

        # Get PRE-state
        pre_state_index = self.current_state_index
        pre_state_int = int(self.ontology[pre_state_index])
        baseline_exon = compute_exon_from_state(pre_state_int)

        # Use orbit representative
        orbit_rep = int(self.phenomenology[pre_state_index])

        # Get token's intron sequence
        introns = self.token_to_introns(token_id)
        if not introns:
            return

        # Evolve through all but last intron
        for intron in introns[:-1]:
            self.current_state_index = self._apply_intron_and_gate(self.current_state_index, intron)

        # BU hinge: learn at closing intron
        closing_intron = introns[-1]
        token_mask = fold(baseline_exon, closing_intron)

        # Update path memory
        self.path_memory = fold(self.path_memory, token_mask)

        # Sparse storage: only if deviates
        if token_mask != baseline_exon:
            if orbit_rep not in self.memory:
                self.memory[orbit_rep] = {}
                self.stats["orbits_discovered"] = len(self.memory)

            # Fold with existing mask
            existing = self.memory[orbit_rep].get(token_id, baseline_exon)
            self.memory[orbit_rep][token_id] = fold(existing, token_mask)

            self.stats["memory_entries"] = sum(len(d) for d in self.memory.values())

        # Apply closing intron
        self.current_state_index = self._apply_intron_and_gate(self.current_state_index, closing_intron)

        # Update stage
        self._update_stage(self.current_state_index)

    def _seed_from_prompt(self, text: str) -> None:
        """Seed state from prompt without learning."""
        tokens = self.tokenizer.encode(text)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids

        for token_id in tokens:
            if token_id in self.valid_tokens:
                introns = self.token_to_introns(token_id)
                for intron in introns:
                    self.current_state_index = self._apply_intron_and_gate(self.current_state_index, intron)
                    self.path_memory = fold(self.path_memory, intron)

        self._update_stage(self.current_state_index)

    def generate_token(self) -> Optional[int]:
        """Generate next token using pure physics."""
        self.stats["generation_steps"] += 1

        # 1. Get candidates from current state's orbit
        orbit_rep = int(self.phenomenology[self.current_state_index])
        candidates = self._orbit_candidates.get(orbit_rep, [])

        if not candidates:
            # Fallback: find nearest populated orbit by theta
            current_theta = float(self.theta[self.current_state_index])
            populated_orbits = {
                orb: self.theta[self.token_post_state_index[cands[0]]]
                for orb, cands in self._orbit_candidates.items()
                if cands
            }
            if not populated_orbits:
                return self.SEP_TOKEN

            best_orb = min(populated_orbits.keys(), key=lambda o: abs(populated_orbits[o] - current_theta))
            candidates = self._orbit_candidates[best_orb]

        if not candidates:
            return self.SEP_TOKEN

        # 2. Calculate Resonance Defect for all candidates
        defects = np.array([self._resonance_defect(c) for c in candidates])
        min_def = min(defects)
        mins = [c for i, c in enumerate(candidates) if defects[i] == min_def]

        # Add diversity: if multiple min, pick via orbit size (larger = more resonant)
        if len(mins) > 1:
            # Filter for actual tokens (not virtual tokens) if multiple candidates
            real_tokens = [t for t in mins if t < self.tokenizer.vocab_size]
            if real_tokens:
                # Use orbit_sizes for real tokens, otherwise fall back to any min
                token = max(real_tokens, key=lambda t: self.orbit_sizes[self.token_post_state_index[t]])
            else:
                token = mins[0]  # Fallback for virtual tokens, pick first
        else:
            token = mins[0]

        if self.debug:
            print(f"  [DBG] Chosen Token: {token} ('{self.tokenizer.decode([token])}')")

        # Apply token's introns
        introns = self.token_to_introns(token)
        for intron in introns:
            self.current_state_index = self._apply_intron_and_gate(self.current_state_index, intron)
            self.path_memory = fold(self.path_memory, intron)

        # Update stage
        self._update_stage(self.current_state_index)

        return token

    def reset(self) -> None:
        """Reset to CS state."""
        self.current_state_index = self.CS_STATE_INDEX
        self.path_memory = GENE_Mic_S
        self.stage_index = 0

        if self.debug:
            theta = float(self.theta[self.CS_STATE_INDEX])
            print(f"🔄 Reset to CS: θ={theta:.4f}")

    def learn_text(self, text: str) -> None:
        """Learn from text."""
        tokens = self.tokenizer.encode(text)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids

        if self.debug:
            print(f"\n📚 Learning {len(tokens)} tokens...")

        initial_memory = self.stats["memory_entries"]

        # Learn CLS token at beginning
        self.learn_token(self.CLS_TOKEN)

        # Learn the text tokens
        for token_id in tokens:
            if token_id in self.valid_tokens:
                self.learn_token(token_id)

        # Learn SEP token at end
        self.learn_token(self.SEP_TOKEN)

        if self.debug:
            new_entries = self.stats["memory_entries"] - initial_memory
            print(f"✓ Learned: {len(tokens)} tokens")
            print(f"  New memory entries: {new_entries}")
            print(f"  Total orbits: {self.stats['orbits_discovered']}")

    def generate_text(self, max_tokens: int = 50) -> str:
        """Generate text."""
        tokens = []

        if self.debug:
            print(f"\n🌀 Generating (max {max_tokens} tokens)...")

        for i in range(max_tokens):
            token_id = self.generate_token()

            if token_id is None:
                if self.debug:
                    print(f"[No resonant token found]")
                break

            # Only decode if it's a real token
            if token_id < self.tokenizer.vocab_size:
                tokens.append(token_id)

                # Stop at IM_END
                if token_id == self.IM_END:
                    if self.debug:
                        print(f"[IM_END token - stopping]")
                    break

        # Decode tokens
        if not tokens:
            return ""

        try:
            text = self.tokenizer.decode(tokens)
        except Exception:
            text = ""

        return text

    def generate_from_prompt(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from prompt."""
        # Reset to clean state
        self.reset()

        # Build proper prompt with SmolLM format
        system_prompt = "You are a helpful AI assistant."
        full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"

        # Seed from prompt (no learning)
        self._seed_from_prompt(full_prompt)

        # Generate response
        return self.generate_text(max_tokens=max_tokens)

    def print_stats(self) -> None:
        """Print kernel statistics."""
        print("\n📊 Statistics:")
        print(f"  Tokens learned: {self.stats['tokens_learned']}")
        print(f"  Memory entries: {self.stats['memory_entries']}")
        print(f"  Orbits discovered: {self.stats['orbits_discovered']}")
        print(f"  Generation steps: {self.stats['generation_steps']}")
        print(f"  Virtual tokens: {len(self.virtual_tokens):,}")

    def run_demo(self):
        """Run demonstration."""
        if self.debug:
            print("\n" + "=" * 60)
            print(f"GyroSI Kernel v0.9.12.0 - {self.model_name} Demo")
            print("=" * 60)

        # Test questions
        test_questions = ["Hello, how are you?", "What is the capital of France?"]

        for question in test_questions:
            if self.debug:
                print(f"\n🤔 User: {question}")
                print(f"🤖 Assistant: ", end="", flush=True)

            response = self.generate_from_prompt(question, max_tokens=50)

            if self.debug:
                print(response)

        if self.debug:
            self.print_stats()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create kernel
    kernel = GyroKernel(debug=True)

    # Run demo
    kernel.run_demo()
