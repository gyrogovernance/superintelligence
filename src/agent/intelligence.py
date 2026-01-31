"""
Gyroscopic ASI Agent.

Integrates:
- Router kernel (CS/UNA: ontology + epistemology)
- Phenomenology operator (ONA: inference interaction)
- Genealogy (BU: input/output cooperation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray

from src.router.kernel import RouterKernel
from src.router.constants import (
    mask12_for_byte,
)

from src.agent.information import (
    horizon_index,
    vertex_charge_for_state,
    K_MIN,
    ETA_DEFAULT,
)
from src.agent.inference import Phenomenology, PhenomenologyState


@dataclass
class AgentConfig:
    """Configuration for Gyroscopic ASI Agent.

K is the number of channels per horizon (D = 256 * K). K must be one of
the supported values listed in src.agent.information.K_VALUES."""
    K: int = K_MIN
    eta: float = ETA_DEFAULT  # Aperture gap
    deterministic: bool = True
    atlas_dir: Path = Path("data/atlas")


@dataclass
class AgentState:
    """Complete agent state (serializable for persistence)."""
    phenomenology: PhenomenologyState
    genealogy: list[int] = field(default_factory=list)
    step: int = 0


class GyroscopicAgent:
    """
    Gyroscopic ASI Agent.
    
    Implements the complete architecture:
    - CS (Ontology): token → byte → state
    - UNA (Epistemology): state transitions
    - ONA (Phenomenology): activation processing, M accumulation, byte selection
    - BU-Eg (Genealogy In): byte logging
    - BU-In (Genealogy Out): byte → token
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
            embedding_fn: function mapping token_id → embedding vector
                         If None, uses zero vectors (for testing)
        """
        self.config = config or AgentConfig()
        
        # Initialise kernel (CS + UNA)
        self.kernel = RouterKernel(self.config.atlas_dir)
        
        # Initialise phenomenology (ONA)
        self.phenomenology = Phenomenology(K=self.config.K, eta=self.config.eta)
        
        # Initialise state
        self.state = AgentState(
            phenomenology=PhenomenologyState.create(self.config.K),
        )
        
        # Embedding function
        self._embedding_fn = embedding_fn or self._zero_embedding
        
        # Track last mask for Hebbian update
        self._last_mask: int = 0
    
    def _zero_embedding(self, token_id: int) -> NDArray[np.float32]:
        """Default embedding: zero vector."""
        return np.zeros(self.phenomenology.D, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.kernel.reset()
        self.state = AgentState(
            phenomenology=PhenomenologyState.create(self.config.K),
        )
        self._last_mask = 0
    
    @property
    def current_state(self) -> int:
        """Current 24-bit kernel state."""
        return int(self.kernel.ontology[self.kernel.state_index])
    
    @property
    def current_horizon(self) -> int:
        """Current horizon index."""
        return horizon_index(self.current_state)
    
    @property
    def current_vertex(self) -> int:
        """Current K₄ vertex charge."""
        return vertex_charge_for_state(self.current_state)
    
    # =========================================================================
    # Token Interface
    # =========================================================================
    
    def token_to_byte(self, token_id: int) -> int:
        """
        Map token ID to byte.
        
        Simple modulo mapping: byte = token_id % 256
        """
        return token_id % 256
    
    def byte_to_tokens(self, byte: int, vocab_size: int = 50000) -> list[int]:
        """
        Map byte to candidate token IDs.
        
        Returns all token_ids where token_id % 256 == byte.
        """
        byte = int(byte) & 0xFF
        return list(range(byte, vocab_size, 256))
    
    def select_token(
        self,
        byte: int,
        logits: Optional[NDArray[np.float32]] = None,
        vocab_size: int = 50000,
    ) -> int:
        """
        Select token from byte using model logits.
        
        Args:
            byte: selected byte
            logits: model output logits (vocab_size,), or None
            vocab_size: vocabulary size
            
        Returns:
            token_id
        """
        candidates = self.byte_to_tokens(byte, vocab_size)
        
        if not candidates:
            # Fallback: return byte directly (for small vocabs)
            return byte
        
        if logits is None:
            # No logits: return first candidate
            return candidates[0]
        
        # Select candidate with highest logit
        candidate_logits = logits[candidates]
        best_idx = int(np.argmax(candidate_logits))
        return candidates[best_idx]
    
    # =========================================================================
    # Core Loop
    # =========================================================================
    
    def step_input(self, token_id: int) -> int:
        """
        Process input token (BU-Eg: Genealogy In).
        
        Args:
            token_id: input token
            
        Returns:
            byte that was processed
        """
        # Token → byte (CS)
        byte = self.token_to_byte(token_id)
        
        # Step kernel (UNA)
        self.kernel.step_byte(byte)
        
        # Record in genealogy
        self.state.genealogy.append(byte)
        self.state.step += 1
        
        # Update last mask for next phenomenology step
        self._last_mask = mask12_for_byte(byte)
        
        return byte
    
    def step_output(
        self,
        embedding: Optional[NDArray[np.float32]] = None,
        token_id: Optional[int] = None,
    ) -> int:
        """
        Generate output byte (ONA + BU-In).
        
        Args:
            embedding: external vector (D,), or None to use embedding_fn
            token_id: if provided, get embedding via embedding_fn
            
        Returns:
            selected byte
        """
        # Get embedding
        if embedding is None:
            if token_id is not None:
                embedding = self._embedding_fn(token_id)
            else:
                embedding = self._embedding_fn(0)  # Default
        
        # Enforce float32 consistency at agent boundary
        embedding = np.asarray(embedding, dtype=np.float32)
        
        # Get current observables
        h_curr = self.current_horizon
        chi_curr = self.current_vertex
        
        # Phenomenology step: update M and select byte
        byte = self.phenomenology.step(
            state=self.state.phenomenology,
            v_curr=embedding,
            h_curr=h_curr,
            delta_mask=self._last_mask,
            chi_curr=chi_curr,
            deterministic=self.config.deterministic,
        )
        
        return byte
    
    def generate(
        self,
        input_tokens: list[int],
        max_output: int = 100,
        get_embedding: Optional[Callable[[int], NDArray[np.float32]]] = None,
        get_logits: Optional[Callable[[list[int]], NDArray[np.float32]]] = None,
        vocab_size: int = 50000,
    ) -> list[int]:
        """
        Generate output tokens from input tokens.
        
        Args:
            input_tokens: input token sequence
            max_output: maximum output tokens
            get_embedding: function to get embedding for token
            get_logits: function to get logits given context
            vocab_size: vocabulary size
            
        Returns:
            output token sequence
        """
        if get_embedding is not None:
            self._embedding_fn = get_embedding
        
        # Process input
        for token_id in input_tokens:
            self.step_input(token_id)
        
        # Generate output
        output_tokens = []
        context = list(input_tokens)
        
        for _ in range(max_output):
            # Get embedding for last token
            last_token = context[-1] if context else 0
            embedding = self._embedding_fn(last_token)
            
            # Get output byte
            byte = self.step_output(embedding=embedding)
            
            # Step kernel with output byte
            self.kernel.step_byte(byte)
            self.state.genealogy.append(byte)
            self.state.step += 1
            self._last_mask = mask12_for_byte(byte)
            
            # Get logits if available
            logits = get_logits(context) if get_logits else None
            
            # Select token
            token = self.select_token(byte, logits, vocab_size)
            output_tokens.append(token)
            context.append(token)
        
        return output_tokens
    
    # =========================================================================
    # Inspection
    # =========================================================================
    
    def get_M(self) -> NDArray[np.float32]:
        """Get current phenomenology field M."""
        return self.state.phenomenology.M.copy()
    
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
            "genealogy_length": len(self.state.genealogy),
            "M_norm": float(np.linalg.norm(self.state.phenomenology.M)),
        }