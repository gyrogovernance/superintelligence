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
from src.agent.adapters import TokenBinding, EmbeddingAdapter, ByteGatingMasks
from src.agent.inference import InferenceFunction, InferenceState


@dataclass
class AgentConfig:
    """Configuration for Gyroscopic ASI Agent.
    
    K is the number of channels per horizon (D = 256 * K). K must be one of
    the supported values listed in src.agent.information.K_VALUES.
    
    vocab_size limits the range of valid token IDs (0 to vocab_size-1).
    Must be in [1, 65536].
    """
    K: int = K_MIN
    eta: float = ETA_DEFAULT
    deterministic: bool = True
    atlas_dir: Path = Path("data/atlas")
    vocab_size: int = 65536
    
    # Optional adapter paths (model-specific, separate from atlas)
    token_binding_path: Optional[Path] = None
    embedding_adapter_path: Optional[Path] = None


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
    - CS (Ontology): token → byte → state
    - UNA (Epistemology): state transitions
    - ONA (Inference): phase-aware activation processing, M accumulation, byte selection
    - BU-Eg (Genealogy In): byte logging
    - BU-In (Genealogy Out): byte → token
    
    Uses palindromic token generation: first byte selects coarse spectral move,
    second byte is chosen with peeked next-state observables (cheaper).
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
        if not (1 <= self.config.vocab_size <= 65536):
            raise ValueError(f"vocab_size must be in [1, 65536], got {self.config.vocab_size}")
        
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
        
        # Initialise state
        self.state = AgentState(
            inference=InferenceState.create(self.config.K),
        )
        
        # Embedding function
        self._embedding_fn = embedding_fn or self._zero_embedding
        
        # Track last mask and vertex for Hebbian update
        self._last_mask: int = 0
        self._last_vertex: int = 0
    
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
        """Convert external token ID to internal 16-bit ID."""
        if self.token_binding is None:
            return int(external_token_id)
        return self.token_binding.to_internal(external_token_id)
    
    def _to_external_id(self, internal_id: int) -> int:
        """Convert internal 16-bit ID to external token ID."""
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
    
    def token_to_bytes(self, token_id: int) -> tuple[int, int]:
        """
        Map 16-bit token to (b1, b2) bytes.
        
        b1 = (token_id >> 8) & 0xFF
        b2 = token_id & 0xFF
        """
        b1 = (token_id >> 8) & 0xFF
        b2 = token_id & 0xFF
        return b1, b2
    
    def bytes_to_token(self, b1: int, b2: int) -> int:
        """
        Reconstruct token from bytes.
        
        token = (b1 << 8) | b2
        """
        return (b1 << 8) | b2
    
    # =========================================================================
    # Core Loop
    # =========================================================================
    
    def step_input(self, token_id: int) -> tuple[int, int]:
        """
        Process input token (BU-Eg: Genealogy In).
        
        Args:
            token_id: input token (external ID)
            
        Returns:
            (b1, b2) bytes that were processed
            
        Raises:
            ValueError: if token_id >= vocab_size
        """
        if token_id >= self.config.vocab_size:
            raise ValueError(f"token_id {token_id} >= vocab_size {self.config.vocab_size}")
        
        # External → internal → bytes
        internal_id = self._to_internal_id(token_id)
        b1, b2 = self.token_to_bytes(internal_id)
        
        for byte in (b1, b2):
            self._step_kernel_and_log(byte)
        
        return b1, b2
    
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
        Generate one token (two kernel steps) using palindromic inference.
        
        First byte selects coarse spectral move.
        Second byte uses peeked next-state observables (cheaper).
        
        Args:
            embedding: input embedding vector (D,)
            
        Returns:
            generated token (external ID)
        """
        X = self.inference.prepare_field(embedding)
        
        h0 = self.kernel.current_horizon
        chi0 = self.kernel.current_vertex
        p0 = self.kernel.current_phase
        delta0 = self._last_mask
        chi_prev0 = self._last_vertex
        
        # Determine gating for b1
        V = self.config.vocab_size
        if self._gating_masks is not None:
            b1_mask = self._gating_masks.allowed_b1
            max_b1 = 255
        else:
            b1_mask = None
            max_b1 = (V - 1) >> 8
        
        b1 = self.inference.step_with_field(
            state=self.state.inference,
            X_curr=X,
            h_curr=h0,
            p_curr=p0,
            delta_mask=delta0,
            chi_prev=chi_prev0,
            chi_curr=chi0,
            deterministic=self.config.deterministic,
            allowed_max_byte=max_b1,
            allowed_mask=b1_mask,
        )
        
        h1 = self.kernel.peek_next_horizon(b1)
        chi1 = self.kernel.peek_next_vertex(b1)
        p1 = self.kernel.peek_next_phase(b1)
        delta1 = mask12_for_byte(b1)
        
        # Determine gating for b2
        if self._gating_masks is not None:
            b2_mask = self._gating_masks.get_b2_mask(b1)
            max_b2 = 255
        else:
            max_b2 = (V - 1) & 0xFF if b1 == (V - 1) >> 8 else 255
            b2_mask = None
        
        b2 = self.inference.step_with_field(
            state=self.state.inference,
            X_curr=X,
            h_curr=h1,
            p_curr=p1,
            delta_mask=delta1,
            chi_prev=chi0,
            chi_curr=chi1,
            deterministic=self.config.deterministic,
            allowed_max_byte=max_b2,
            allowed_mask=b2_mask,
        )
        
        self._step_kernel_and_log(b1)
        self._step_kernel_and_log(b2)
        
        internal_id = self.bytes_to_token(b1, b2)
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