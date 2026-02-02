"""
Gyroscopic ASI Agent.

Integrates:
- Router kernel (CS/UNA: ontology + epistemology + spectral atlas)
- Inference Function (ONA: phase-aware inference)
- Genealogy (BU: input/output cooperation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray

from src.router.kernel import RouterKernel
from src.router.constants import mask12_for_byte, GENE_MIC_S

from src.agent.information import (
    K_MIN,
    K_VALUES,
    ETA_DEFAULT,
)
from src.agent.adapters import TokenBinding, EmbeddingAdapter, ByteGatingMasks, SemanticTokenCodec
from src.agent.inference import InferenceFunction, InferenceState


@dataclass
class AgentConfig:
    """Configuration for Gyroscopic ASI Agent.
    
    K is the number of channels per horizon (D = 256 * K). K must be one of
    the supported values listed in src.agent.information.K_VALUES.
    
    vocab_size limits the range of valid token IDs (0 to vocab_size-1).
    Must be in [1, 2^(8*bytes_per_token)].
    
    bytes_per_token is the fixed width (L) for token-to-byte encoding.
    Default is 2 (legacy 16-bit). Use 4 for models like OLMo with >65536 vocab.
    
    semantic_codec_path: if provided, uses LSH-based semantic encoding/decoding
    for tokens instead of raw big-endian byte mapping. Required for meaningful
    language generation with large vocabularies.
    """
    K: int = K_MIN
    eta: float = ETA_DEFAULT
    deterministic: bool = True
    atlas_dir: Path = Path("data/atlas")
    vocab_size: int = 65536
    bytes_per_token: int = 2  # L: 2 for legacy 16-bit, 4 for OLMo (32-bit)
    
    # Optional adapter paths (model-specific, separate from atlas)
    token_binding_path: Optional[Path] = None
    embedding_adapter_path: Optional[Path] = None
    semantic_codec_path: Optional[Path] = None  # SemanticTokenCodec for 4-byte LSH


@dataclass
class AgentState:
    """Complete agent state (serializable for persistence)."""
    inference: InferenceState
    genealogy: list[int] = field(default_factory=list)
    step: int = 0


class GyroscopicAgent:
    """
    Gyroscopic ASI Agent.
    
    Implements the complete architecture:
    - CS (Ontology): token -> byte -> state
    - UNA (Epistemology): state transitions
    - ONA (Inference): phase-aware activation processing, M accumulation, byte selection
    - BU-Eg (Genealogy In): byte logging
    - BU-In (Genealogy Out): byte -> token
    
    Uses sequential prefix-peek planning for token generation:
    pick b1, peek -> pick b2, peek -> ... -> pick bL (for L = bytes_per_token).
    Each byte is chosen with vocab-gated constraints and peeked next-state observables.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        embedding_fn: Optional[Callable[[int], NDArray[np.float32]]] = None,
    ):
        """
        Initialise agent.
        
        Args:
            config: agent configuration
            embedding_fn: function mapping context ID to embedding vector
                         If None, uses zero vectors (for testing)
        """
        self.config = config or AgentConfig()
        
        # Validate config
        if self.config.K not in K_VALUES:
            raise ValueError(f"K={self.config.K} is not supported. Supported: {K_VALUES}")
        
        L = self.config.bytes_per_token
        if not (1 <= L <= 8):
            raise ValueError(f"bytes_per_token must be in [1, 8], got {L}")
        
        max_vocab = 1 << (8 * L)
        if not (1 <= self.config.vocab_size <= max_vocab):
            raise ValueError(
                f"vocab_size must be in [1, {max_vocab}] for bytes_per_token={L}, "
                f"got {self.config.vocab_size}"
            )
        
        # Initialise kernel (CS + UNA + spectral atlas)
        self.kernel = RouterKernel(self.config.atlas_dir)
        
        # Initialise inference function (ONA)
        self.inference = InferenceFunction(K=self.config.K, eta=self.config.eta)
        
        # Load precomputed tables from kernel's phenomenology
        feature_key = f"features_K{self.config.K}"
        with np.load(self.config.atlas_dir / "phenomenology.npz", allow_pickle=False) as phen:
            if feature_key not in phen.files:
                raise ValueError(
                    f"phenomenology.npz missing '{feature_key}'. Rebuild atlas with: python -m src.router.atlas"
                )

            self.inference.set_kernel_tables(
                byte_weight=self.kernel.byte_weight,
                byte_charge=self.kernel.byte_charge,
                byte_features=phen[feature_key],
                gamma_table=self.kernel.gamma_table,
            )
        
        # Optional adapters
        self.token_binding: Optional[TokenBinding] = None
        self.embedding_adapter: Optional[EmbeddingAdapter] = None
        self.semantic_codec: Optional[SemanticTokenCodec] = None
        self._gating_masks: Optional[ByteGatingMasks] = None
        
        if self.config.token_binding_path is not None:
            self.token_binding = TokenBinding.load(self.config.token_binding_path)
            if self.token_binding.vocab_size != self.config.vocab_size:
                raise ValueError(
                    f"TokenBinding vocab_size={self.token_binding.vocab_size} != "
                    f"AgentConfig vocab_size={self.config.vocab_size}"
                )
            self._gating_masks = ByteGatingMasks.from_token_binding(self.token_binding)
        
        if self.config.embedding_adapter_path is not None:
            self.embedding_adapter = EmbeddingAdapter.load(self.config.embedding_adapter_path)
            if self.embedding_adapter.output_dim != self.inference.D:
                raise ValueError(
                    f"EmbeddingAdapter output_dim={self.embedding_adapter.output_dim} != "
                    f"gyro D={self.inference.D}"
                )
        
        if self.config.semantic_codec_path is not None:
            self.semantic_codec = SemanticTokenCodec.load(self.config.semantic_codec_path)
            if self.semantic_codec.vocab_size != self.config.vocab_size:
                raise ValueError(
                    f"SemanticTokenCodec vocab_size={self.semantic_codec.vocab_size} != "
                    f"AgentConfig vocab_size={self.config.vocab_size}"
                )
        
        # Initialise state
        self.state = AgentState(
            inference=InferenceState.create(self.config.K),
        )
        
        # Embedding function
        self._embedding_fn = embedding_fn or self._zero_embedding
        
        # Track last mask and vertex for Hebbian update (from kernel's initial state)
        self._last_mask: int = mask12_for_byte(self.kernel.last_byte)
        self._last_vertex: int = self.kernel.current_vertex
    
    def _zero_embedding(self, token_id: int) -> NDArray[np.float32]:
        """Default embedding: zero vector."""
        return np.zeros(self.inference.D, dtype=np.float32)
    
    def _step_kernel_and_log(self, byte: int) -> None:
        """Step kernel and log genealogy with correct vertex tracking."""
        prev_vertex = self.kernel.current_vertex  # vertex BEFORE stepping
        self.kernel.step_byte(byte)

        self.state.genealogy.append(int(byte) & 0xFF)
        self.state.step += 1

        self._last_mask = mask12_for_byte(byte)
        self._last_vertex = prev_vertex
    
    def _to_internal_id(self, external_token_id: int) -> int:
        """Convert external token ID to internal ID."""
        if self.token_binding is None:
            return int(external_token_id)
        return self.token_binding.to_internal(external_token_id)
    
    def _to_external_id(self, internal_id: int) -> int:
        """Convert internal ID to external token ID."""
        if self.token_binding is None:
            return int(internal_id)
        return self.token_binding.to_external(internal_id)
    
    def _ensure_gyro_embedding(self, emb: NDArray[np.float32]) -> NDArray[np.float32]:
        """Ensure embedding has correct gyro dimension D."""
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] == self.inference.D:
            return emb
        if self.embedding_adapter is not None and emb.shape[0] == self.embedding_adapter.input_dim:
            return self.embedding_adapter.project(emb)
        raise ValueError(
            f"Embedding dimension {emb.shape[0]} != gyro D={self.inference.D}"
            + (f" and != adapter input_dim={self.embedding_adapter.input_dim}" 
               if self.embedding_adapter else "")
        )
    
    def _allowed_max_next_byte_contiguous(self, prefix: list[int]) -> int:
        """
        For contiguous vocab [0..V-1], big-endian fixed width L.
        
        Returns max allowed byte at the next position, assuming prefix is valid.
        If prefix already matches the max token's prefix, the next byte is
        capped at the corresponding byte of max_id. Otherwise, returns 255.
        """
        L = self.config.bytes_per_token
        V = self.config.vocab_size
        max_id = V - 1
        max_bytes = list(max_id.to_bytes(L, "big"))
        
        i = len(prefix)
        if i >= L:
            raise ValueError("prefix already complete")
        
        # Compare prefix with max_bytes prefix
        for j in range(i):
            if prefix[j] < max_bytes[j]:
                return 255  # Unconstrained
            if prefix[j] > max_bytes[j]:
                raise ValueError("prefix exceeds vocab range")
        # Equal so far: cap at max_bytes[i]
        return max_bytes[i]
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.kernel.reset()
        self.state = AgentState(
            inference=InferenceState.create(self.config.K),
        )
        self._last_mask = mask12_for_byte(GENE_MIC_S)
        self._last_vertex = self.kernel.current_vertex
    
    @property
    def current_state(self) -> int:
        """Current 24-bit kernel state."""
        return int(self.kernel.ontology[self.kernel.state_index])
    
    @property
    def current_horizon(self) -> int:
        """Current horizon index (from kernel's spectral atlas)."""
        return self.kernel.current_horizon
    
    @property
    def current_vertex(self) -> int:
        """Current K₄ vertex charge (from kernel's spectral atlas)."""
        return self.kernel.current_vertex
    
    @property
    def current_phase(self) -> int:
        """Current phase in permutation cycle."""
        return self.kernel.current_phase
    
    # =========================================================================
    # Token Interface
    # =========================================================================
    
    def token_to_bytes(self, token_id: int) -> list[int]:
        """
        Map token to L bytes.
        
        If semantic_codec is loaded, uses LSH-based semantic encoding.
        Otherwise uses big-endian fixed-width encoding.
        
        L = bytes_per_token. Returns list of L bytes.
        """
        if self.semantic_codec is not None:
            return self.semantic_codec.encode(token_id)
        
        L = self.config.bytes_per_token
        max_id = (1 << (8 * L)) - 1
        if not (0 <= token_id <= max_id):
            raise ValueError(
                f"token_id {token_id} out of range for bytes_per_token={L}"
            )
        return list(int(token_id).to_bytes(L, "big"))
    
    def bytes_to_token(self, bs: list[int]) -> int:
        """
        Reconstruct token from L bytes (big-endian).
        
        Note: For semantic codec, use bytes_to_token_semantic() instead,
        which requires probe and embed_tokens for proper decoding.
        
        L = bytes_per_token. bs must have exactly L elements.
        """
        L = self.config.bytes_per_token
        if len(bs) != L:
            raise ValueError(f"expected {L} bytes, got {len(bs)}")
        return int.from_bytes(bytes(int(b) & 0xFF for b in bs), "big")
    
    def bytes_to_token_semantic(
        self,
        bs: list[int],
        probe: "NDArray[np.float32]",
        embed_tokens: "NDArray[np.float32]",
    ) -> int:
        """
        Decode bytes to token using semantic codec with prefix backoff.
        
        Args:
            bs: L bytes from byte selection
            probe: context probe vector for scoring candidates
            embed_tokens: embedding table for candidate scoring
            
        Returns:
            token_id decoded with semantic prefix backoff
        """
        if self.semantic_codec is None:
            return self.bytes_to_token(bs)
        
        import torch
        probe_t = torch.from_numpy(probe).float()
        embed_t = torch.from_numpy(embed_tokens)
        return self.semantic_codec.decode(bs, probe_t, embed_t)
    
    # =========================================================================
    # Core Loop
    # =========================================================================
    
    def step_input(self, token_id: int) -> list[int]:
        """
        Process input token (BU-Eg: Genealogy In).
        
        Args:
            token_id: input token (external ID)
            
        Returns:
            list of L bytes that were processed
            
        Raises:
            ValueError: if token_id >= vocab_size
        """
        if token_id >= self.config.vocab_size:
            raise ValueError(f"token_id {token_id} >= vocab_size {self.config.vocab_size}")
        
        # External -> internal -> bytes
        internal_id = self._to_internal_id(token_id)
        bs = self.token_to_bytes(internal_id)
        
        for byte in bs:
            self._step_kernel_and_log(byte)
        
        return bs
    
    def step_output(
        self,
        embedding: Optional[NDArray[np.float32]] = None,
        token_id: Optional[int] = None,
    ) -> int:
        """
        Generate output token (ONA + BU-In).
        
        Args:
            embedding: external vector, or None to use embedding_fn
            token_id: if provided, get embedding via embedding_fn
            
        Returns:
            selected token (external ID)
        """
        if embedding is None:
            embedding = self._embedding_fn(token_id if token_id is not None else 0)
        
        gyro_emb = self._ensure_gyro_embedding(embedding)
        return self.generate_token(gyro_emb)
    
    def generate(
        self,
        input_tokens: list[int],
        max_output: int = 100,
        get_embedding: Optional[Callable[[int], NDArray[np.float32]]] = None,
    ) -> list[int]:
        """
        Generate output tokens from input tokens.
        
        Args:
            input_tokens: input token sequence
            max_output: maximum output tokens
            get_embedding: function to get embedding for token (unused)
            
        Returns:
            output token sequence
        """
        if get_embedding is not None:
            self._embedding_fn = get_embedding
        
        effective_vocab = self.config.vocab_size
        
        # Process input
        for token_id in input_tokens:
            if token_id < effective_vocab:
                self.step_input(token_id)
        
        # Generate output
        output_tokens = []
        context = list(input_tokens)
        
        for _ in range(max_output):
            last_token = context[-1] if context else 0
            emb = self._embedding_fn(last_token)
            gyro_emb = self._ensure_gyro_embedding(emb)
            
            token = self.generate_token(gyro_emb)
            
            # Ensure token is valid
            if token >= effective_vocab:
                raise RuntimeError(f"Generated out-of-range token {token} for vocab {effective_vocab}")
            
            output_tokens.append(token)
            context.append(token)
        
        return output_tokens
    
    def generate_token(self, embedding: NDArray[np.float32]) -> int:
        """
        Generate one token (L kernel steps) using sequential prefix-peek planning.
        
        For each byte position i in [0..L-1]:
        1. Read spectral observables (h, chi, p) from ephemeral state
        2. Select byte via Inference Function with vocab-gated constraint
        3. Advance ephemeral state via epistemology lookup (peek)
        
        After all L bytes are planned, commit them to the real kernel and genealogy.
        
        Args:
            embedding: input embedding vector (D,)
            
        Returns:
            generated token (external ID)
        """
        X = self.inference.prepare_field(embedding)
        L = self.config.bytes_per_token
        
        # Ephemeral planning state (starts from real kernel state)
        idx = int(self.kernel.state_index)
        last_byte = int(self.kernel.last_byte)
        delta_mask = int(self._last_mask)
        chi_prev = int(self._last_vertex)
        
        planned: list[int] = []
        
        for _ in range(L):
            # Read spectral observables from ephemeral state
            h = int(self.kernel.state_horizon[idx])
            chi = int(self.kernel.state_vertex[idx])
            p = int(self.kernel.phase[idx, last_byte])
            
            # Vocab gating for this byte position
            allowed_max = self._allowed_max_next_byte_contiguous(planned)
            
            # Select byte via inference
            b = self.inference.step_with_field(
                state=self.state.inference,
                X_curr=X,
                h_curr=h,
                p_curr=p,
                delta_mask=delta_mask,
                chi_prev=chi_prev,
                chi_curr=chi,
                deterministic=self.config.deterministic,
                allowed_max_byte=allowed_max,
                allowed_mask=None,
            )
            
            planned.append(b)
            
            # Advance ephemeral state by chosen byte (peek, no commit)
            idx = int(self.kernel.epistemology[idx, b])
            last_byte = int(b)
            delta_mask = mask12_for_byte(b)
            chi_prev = chi
        
        # Commit: step the real kernel and log genealogy
        for b in planned:
            self._step_kernel_and_log(b)
        
        internal_id = self.bytes_to_token(planned)
        
        # Safety check
        if internal_id >= self.config.vocab_size:
            raise RuntimeError(
                f"planned bytes produced out-of-range token_id {internal_id} "
                f"(vocab_size={self.config.vocab_size})"
            )
        
        return self._to_external_id(internal_id)
    
    # =========================================================================
    # Inspection
    # =========================================================================
    
    def get_M(self) -> NDArray[np.float32]:
        """Get current inference field M (256, 4, K)."""
        return self.state.inference.M.copy()
    
    def get_genealogy(self) -> list[int]:
        """Get byte genealogy."""
        return list(self.state.genealogy)
    
    def get_signature(self) -> dict[str, int | str | float]:
        """Get current state signature."""
        sig = self.kernel.signature()
        return {
            "step": self.state.step,
            "kernel_step": sig.step,
            "state_hex": sig.state_hex,
            "horizon": self.current_horizon,
            "vertex": self.current_vertex,
            "phase": self.current_phase,
            "genealogy_length": len(self.state.genealogy),
            "M_norm": float(np.linalg.norm(self.state.inference.M)),
        }