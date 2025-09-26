"""
GyroSI Kernel v0.9.7.2 - Pure Physics Implementation

This implements the Common Governance Model (CGM) through a physics-first approach
to language processing. The kernel uses all five physics maps:

1. Ontology: The 789,170 states that form the finite, closed state manifold
2. Epistemology: State transition table (789,170 × 256) mapping (state, intron) → state
3. Theta: Angular divergence from archetype, measuring position in the CGM cycle
4. Phenomenology: Maps states to their canonical orbit representatives (256 orbits)
5. Orbit Sizes: Cardinality of each phenomenological orbit

The system implements the CGM 8-fold path through recursive alignment:
- CS (Common Source): Unobservable origin with inherent chirality
- UNA (Unity Non-Absolute): First observable structure with non-identity right gyration
- ONA (Opposition Non-Absolute): Full differentiation with maximal non-associativity
- BU_EG (Balance Universal - Egress): Expression of accumulated intelligence
- BU_IN (Balance Universal - Ingress): Integration of experience via Monodromic Fold
- ONA (return): Mediated opposition in return path
- UNA (return): Non-absolute unity in return path
- CS (closure): Return to source, completing the helical cycle

Pure physics implementation with holographic compression and minimal complexity.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path


# Core Constants
GENE_Mic_S = 0xAA  # Holographic topology constant (ψ seed)

# GENE_Mac_S: 48-bit archetypal tensor [4, 2, 3, 2]
GENE_Mac_S = np.array(
    [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ],
    dtype=np.int8,
)

# Bit family masks per CGM
EXON_FG_MASK = 0b00100100  # Bits 2, 5 - ONA (Forward Gyration)
EXON_BG_MASK = 0b00011000  # Bits 3, 4 - BU (Backward Gyration)

# Special tokens
CLS_TOKEN = 101
SEP_TOKEN = 102
PAD_TOKEN = 0

# CS State constant (state integer 0)
CS_STATE_INT = 0


def tensor_to_int(tensor: np.ndarray) -> int:
    """Convert 48-bit tensor to integer state."""
    if tensor.shape != (4, 2, 3, 2):
        raise ValueError("Expected tensor shape (4, 2, 3, 2), got {tensor.shape}")
    bits = (tensor.flatten(order="C") == -1).astype(np.uint8)
    result = 0
    for i, bit in enumerate(bits):
        if bit:
            result |= 1 << i
    return result


def fold(a: int, b: int) -> int:
    """
    The Monodromic Fold (⋄), the path-dependent learning operator.

    Canonical Form: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
    Algebraic Normal Form: a ⋄ b = ¬a ∧ b

    These are mathematically identical through Boolean algebra.
    """
    a &= 0xFF
    b &= 0xFF
    negated_b = (~b) & 0xFF
    gyration = b ^ (a & negated_b)
    return (a ^ gyration) & 0xFF


def fold_sequence(values: List[int], start_state: int = 0) -> int:
    """Apply Monodromic Fold left-to-right over a sequence."""
    result = start_state & 0xFF
    for value in values:
        result = fold(result, value & 0xFF)
    return result


def compute_exon_product(state_int: int) -> int:
    """Compute exon product from state using first 8 bits."""
    return (state_int & 0xFF) if state_int != 0 else GENE_Mic_S


def transcribe_byte(byte: int) -> int:
    """ψ isomorphism: byte → intron via XOR 0xAA."""
    return (byte ^ GENE_Mic_S) & 0xFF


def token_id_to_leb128(token_id: int) -> List[int]:
    """Convert token ID to LEB128 bytes."""
    if token_id < 0:
        raise ValueError("Token ID must be non-negative")
    bytes_list = []
    while True:
        byte = token_id & 0x7F
        token_id >>= 7
        if token_id == 0:
            bytes_list.append(byte)
            break
        else:
            bytes_list.append(byte | 0x80)
    return bytes_list


def token_to_introns(token_id: int) -> List[int]:
    """Convert token ID to intron sequence via ψ isomorphism."""
    leb_bytes = token_id_to_leb128(token_id)
    return [transcribe_byte(b) for b in leb_bytes]


# Precompute intron broadcast masks for CS emission
INTRON_BROADCAST_MASKS = np.zeros(256, dtype=np.uint64)
for intron in range(256):
    # Simple Traceable mapping for CS emission
    # Each intron creates a unique state pattern
    mask = (intron * 0x9E3779B97F4A7C15) & ((1 << 48) - 1)
    INTRON_BROADCAST_MASKS[intron] = mask


class GyroKernel:
    """GyroSI Kernel v0.9.7.2 - Simplified Pure Physics Implementation

    Core principles:
    - State evolution via epistemology table
    - Learning via Monodromic Fold
    - Holographic compression (store only deltas)
    - Generation via fold defect minimization
    - No arbitrary rules or thresholds
    """

    def __init__(self, base_path: Optional[Path] = None, verbose: bool = True):
        """Initialize kernel with physics tables."""
        if base_path is None:
            base_path = Path(__file__).parents[1] / "memories"

        self.base_path = base_path
        self.verbose = verbose

        # Load physics tables
        self._load_complete_physics()

        # CS is always state 0
        self.cs_index = 0
        self.current_state_index = self.cs_index

        # Build orbit mapping
        unique_reps = np.unique(self.phenomenology)
        self.rep_to_orbit_id: Dict[int, int] = {int(rep): int(i) for i, rep in enumerate(unique_reps)}

        # Holographic storage: orbit -> token -> delta (compressed)
        self.orbit_deltas: Dict[int, Dict[int, int]] = {}

        # Per-orbit path memory for local context
        self.orbit_memories: Dict[int, int] = {}

        # Context window (6-bit diameter)
        self.context_window: List[int] = []
        self.window_size = 6

        # Valid tokens
        self.valid_tokens: Set[int] = set()

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for GyroSI physics")
        self._build_token_structures()

        if self.verbose:
            cs_theta = self._get_theta(self.cs_index)
            print("GyroSI Kernel v0.9.7.2 initialized")
            print("CS state: index={self.cs_index}, θ={cs_theta:.3f}")

    def _load_complete_physics(self) -> None:
        """Load ALL physics tables."""
        meta_path = self.base_path / "public" / "meta"

        # Load ontology (state integers)
        ontology_path = meta_path / "ontology_keys.npy"
        if not ontology_path.exists():
            raise FileNotFoundError("Ontology not found: {ontology_path}")
        self.ontology = np.load(ontology_path, mmap_mode="r")

        # Load epistemology (state transition table)
        epistemology_path = meta_path / "epistemology.npy"
        if not epistemology_path.exists():
            raise FileNotFoundError("Epistemology not found: {epistemology_path}")
        self.epistemology = np.load(epistemology_path, mmap_mode="r")

        # Load theta (angular divergence from archetype)
        theta_path = meta_path / "theta.npy"
        if not theta_path.exists():
            raise FileNotFoundError("Theta not found: {theta_path}")
        self.theta = np.load(theta_path, mmap_mode="r")

        # Load phenomenology map (state -> canonical orbit representative)
        pheno_path = meta_path / "phenomenology_map.npy"
        if not pheno_path.exists():
            raise FileNotFoundError("Phenomenology not found: {pheno_path}")
        self.phenomenology = np.load(pheno_path, mmap_mode="r")

        # Load orbit sizes
        orbit_sizes_path = meta_path / "orbit_sizes.npy"
        if not orbit_sizes_path.exists():
            raise FileNotFoundError("Orbit sizes not found: {orbit_sizes_path}")
        self.orbit_sizes = np.load(orbit_sizes_path, mmap_mode="r")

    def _load_tokenizer(self):
        """Load the tokenizer as a physics component."""
        try:
            from tokenizers import Tokenizer

            tokenizer_path = self.base_path / "public" / "tokenizers"
            tokenizer_path = tokenizer_path / "bert-base-uncased" / "tokenizer.json"
            if tokenizer_path.exists():
                return Tokenizer.from_file(str(tokenizer_path))
        except ImportError:
            pass
        return None

    def _build_token_structures(self) -> None:
        """Build valid token set."""
        vocab_size = self.tokenizer.get_vocab_size()
        for token_id in range(min(vocab_size, 30000)):
            token = self.tokenizer.id_to_token(token_id)
            if token and not token.startswith("[unused"):
                self.valid_tokens.add(token_id)

    def _get_state_index(self, state_int: int) -> int:
        """Get ontology index for a state integer."""
        idx = np.searchsorted(self.ontology, state_int)
        if idx < len(self.ontology) and self.ontology[idx] == state_int:
            return int(idx)
        return self.cs_index

    def _get_state_int(self, state_index: int) -> int:
        """Get state integer from ontology index."""
        if 0 <= state_index < len(self.ontology):
            return int(self.ontology[state_index])
        return CS_STATE_INT

    def _get_theta(self, state_index: int) -> float:
        """Get theta (angular divergence) for a state."""
        if 0 <= state_index < len(self.theta):
            return float(self.theta[state_index])
        return 0.0

    def _get_orbit_rep(self, state_index: int) -> int:
        """Get canonical orbit representative for a state."""
        if 0 <= state_index < len(self.phenomenology):
            return int(self.phenomenology[state_index])
        return state_index

    def _get_orbit_size(self, state_index: int) -> int:
        """Get size of an orbit."""
        if 0 <= state_index < len(self.orbit_sizes):
            return int(self.orbit_sizes[state_index])
        return 1

    def apply_gyration_and_transform(self, state_int: int, intron: int) -> int:
        """Apply intron to state using generic physics."""
        intron &= 0xFF

        # Generic transition via epistemology for all states
        state_index = self._get_state_index(state_int)
        next_index = self.epistemology[state_index, intron]
        return self._get_state_int(next_index)

    def process_token(self, token_id: int) -> None:
        """Learn a token using physics-based memory and Monodromic Fold."""
        if token_id not in self.valid_tokens:
            return

        introns = token_to_introns(token_id)
        if not introns:
            return

        # Track PRE-state for learning
        pre_state_index = self.current_state_index
        pre_orbit_rep = self._get_orbit_rep(pre_state_index)
        pre_state_int = self._get_state_int(pre_state_index)
        baseline_exon = compute_exon_product(pre_state_int)

        # Evolve through all introns
        for intron in introns:
            next_state_int = self.apply_gyration_and_transform(self._get_state_int(self.current_state_index), intron)
            self.current_state_index = self._get_state_index(next_state_int)

            # Update context window
            self.context_window.append(intron)
            if len(self.context_window) > self.window_size:
                self.context_window.pop(0)

        # Compute token mask using fold with context
        context_fold = fold_sequence(self.context_window, start_state=0)
        token_mask = fold_sequence(introns, start_state=context_fold)

        # HOLOGRAPHIC COMPRESSION: Store only delta from baseline
        delta = token_mask ^ baseline_exon
        if delta != 0:  # Only store if different from baseline
            if pre_orbit_rep not in self.orbit_deltas:
                self.orbit_deltas[pre_orbit_rep] = {}

            # Update or store delta
            existing_delta = self.orbit_deltas[pre_orbit_rep].get(token_id, 0)
            new_delta = fold(existing_delta, delta)
            self.orbit_deltas[pre_orbit_rep][token_id] = new_delta

        # Update per-orbit path memory
        if pre_orbit_rep not in self.orbit_memories:
            self.orbit_memories[pre_orbit_rep] = GENE_Mic_S
        self.orbit_memories[pre_orbit_rep] = fold(self.orbit_memories[pre_orbit_rep], token_mask)

    def generate_token(self) -> int:
        """Generate token via pure fold-based resonance."""
        current_orbit_rep = self._get_orbit_rep(self.current_state_index)
        current_state_int = self._get_state_int(self.current_state_index)
        baseline_exon = compute_exon_product(current_state_int)

        # Get orbit memory or use global seed
        orbit_memory = self.orbit_memories.get(current_orbit_rep, GENE_Mic_S)

        # Compute context defect
        context_fold = fold_sequence(self.context_window, start_state=0)
        path_defect = fold(orbit_memory, context_fold)

        # Find token with minimum fold defect
        best_token = None
        min_defect = 256  # Max possible defect

        # Check patterns in current orbit
        if current_orbit_rep in self.orbit_deltas:
            for token_id, delta in self.orbit_deltas[current_orbit_rep].items():
                # Reconstruct actual mask from delta
                mask = delta ^ baseline_exon

                # Compute defect as hamming weight of fold result
                defect_value = fold(path_defect, mask)
                defect = bin(defect_value).count("1")

                if defect < min_defect:
                    min_defect = defect
                    best_token = token_id

        # If no patterns in current orbit, check nearby orbits by defect
        if best_token is None:
            orbit_size = self._get_orbit_size(self.current_state_index)

            # Use orbit size to determine search range
            search_range = max(1, 256 // max(1, orbit_size))

            for orbit_rep in list(self.orbit_deltas.keys())[:search_range]:
                if orbit_rep == current_orbit_rep:
                    continue

                for token_id, delta in self.orbit_deltas[orbit_rep].items():
                    # Reconstruct mask using current baseline
                    mask = delta ^ baseline_exon

                    defect_value = fold(path_defect, mask)
                    defect = bin(defect_value).count("1")

                    if defect < min_defect:
                        min_defect = defect
                        best_token = token_id

        # If still no token, try special tokens based on theta
        if best_token is None:
            current_theta = self._get_theta(self.current_state_index)

            # Near CS, try CLS
            if current_theta > 1.5 and CLS_TOKEN in self.valid_tokens:
                best_token = CLS_TOKEN
            # Near closure, try SEP
            elif current_theta < 0.2 and SEP_TOKEN in self.valid_tokens:
                best_token = SEP_TOKEN
            # Otherwise, any valid token
            else:
                for token_id in self.valid_tokens:
                    if token_id not in [CLS_TOKEN, SEP_TOKEN, PAD_TOKEN]:
                        best_token = token_id
                        break

        return best_token if best_token is not None else PAD_TOKEN

    def _evolve_state(self, token_id: int) -> None:
        """Evolve state without learning (for generation)."""
        introns = token_to_introns(token_id)
        for intron in introns:
            next_state_int = self.apply_gyration_and_transform(self._get_state_int(self.current_state_index), intron)
            self.current_state_index = self._get_state_index(next_state_int)

            # Update context window
            self.context_window.append(intron)
            if len(self.context_window) > self.window_size:
                self.context_window.pop(0)

    def reset_to_cs(self) -> None:
        """Reset to CS state as extra-phenomenal reference point."""
        self.current_state_index = self.cs_index
        self.context_window = []

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        encoding = self.tokenizer.encode(text)
        return [t for t in encoding.ids if t in self.valid_tokens]

    def tokens_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs to text."""
        filtered = [t for t in token_ids if t not in [CLS_TOKEN, SEP_TOKEN, PAD_TOKEN]]
        if not filtered:
            return ""
        try:
            return self.tokenizer.decode(filtered)
        except:
            return " ".join("[{t}]" for t in filtered)

    def generate_text(self, max_tokens: int = 50, debug: bool = False, prompt: Optional[str] = None) -> str:
        """Generate text using pure physics-based resonance."""
        # Process prompt to evolve state and learn
        if prompt:
            prompt_tokens = self.text_to_tokens(prompt)
            if prompt_tokens:
                for token_id in prompt_tokens:
                    self.process_token(token_id)

                if debug:
                    print("Processed prompt: {len(prompt_tokens)} tokens")
                    print("After prompt: θ={self._get_theta(self.current_state_index):.3f}")
                    print("Learned orbits: {len(self.orbit_deltas)}")
                    total_deltas = sum(len(d) for d in self.orbit_deltas.values())
                    print("Total deltas stored: {total_deltas}")

        tokens = []

        if debug:
            print("\n{'=' * 50}")
            print("GENERATION")
            print("Initial: θ={self._get_theta(self.current_state_index):.3f}")
            print("{'=' * 50}")

        for i in range(max_tokens):
            token_id = self.generate_token()

            # Stop at PAD token
            if token_id == PAD_TOKEN:
                if debug:
                    print("[No resonance found]")
                break

            tokens.append(token_id)

            # Get token text for debug
            if debug:
                token_text = self.tokens_to_text([token_id])
                if not token_text:
                    if token_id == CLS_TOKEN:
                        token_text = "[CLS]"
                    elif token_id == SEP_TOKEN:
                        token_text = "[SEP]"
                    else:
                        token_text = "[{token_id}]"

                # Update state for next generation
                old_theta = self._get_theta(self.current_state_index)
                self._evolve_state(token_id)
                new_theta = self._get_theta(self.current_state_index)

                theta_change = "↑" if new_theta > old_theta else "↓" if new_theta < old_theta else "→"
                print("  {i+1:2d}: '{token_text}' → θ={new_theta:.3f} {theta_change}")
            else:
                self._evolve_state(token_id)

            # Natural stopping at SEP
            if token_id == SEP_TOKEN and i > 3:
                if debug:
                    print("[Natural ending at SEP]")
                break

        result = self.tokens_to_text(tokens)

        if debug:
            print("\nGenerated: {result}")
            print("{'=' * 50}\n")

        return result


def demo_kernel():
    """GyroSI Kernel v0.9.7.2 Demonstration"""
    print("\nGyroSI Kernel v0.9.7.2 - Simplified Pure Physics")
    print("=" * 60)

    # Create kernel
    kernel = GyroKernel(verbose=True)

    # Load wiki text as prompt
    wiki_file = "toys/training/wiki_test.txt"
    try:
        with open(wiki_file, "r", encoding="utf-8") as f:
            wiki_text = f.read()
        sample_text = wiki_text[:100] + "..." if len(wiki_text) > 100 else wiki_text
        print("\nWiki text prompt (sample): '{sample_text}'")
    except FileNotFoundError:
        wiki_text = "In mathematics and computer science, an algorithm is a sequence of instructions."
        print("\nUsing fallback prompt: '{wiki_text}'")

    # Generate response
    print("\nGENERATION TEST:")
    kernel.reset()
    generated = kernel.generate_text(max_tokens=30, debug=True, prompt=wiki_text)

    print("\nDEMONSTRATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo_kernel()
