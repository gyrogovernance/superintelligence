"""
Phenomenology operator.

Implements:
- Reshape: external vector → horizon × channel tensor
- Hebbian update: order-sensitive accumulation in M
- Byte scoring: deterministic selection using kernel observables
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from src.agent.information import (
    ETA_DEFAULT,
    K_MIN,
    direction_factor,
    mask_weight,
    vertex_charge_for_byte,
    byte_feature_matrix,
    M_CLIP,
)


@dataclass
class PhenomenologyState:
    """
    Accumulated phenomenology state.
    
    M: (256, K) horizon × channel field
    h_prev: previous horizon index
    a_prev: previous local activation
    """
    M: NDArray[np.float32]
    h_prev: int = 0
    a_prev: Optional[NDArray[np.float32]] = None
    
    @classmethod
    def create(cls, K: int) -> "PhenomenologyState":
        """Initialize with zero M."""
        return cls(
            M=np.zeros((256, K), dtype=np.float32),
            h_prev=0,
            a_prev=None,
        )


class Phenomenology:
    """
    Phenomenology operator for Gyroscopic ASI.
    
    Maps external vectors + kernel state → byte selection,
    accumulating trajectory-dependent structure in M.
    """
    
    def __init__(self, K: int = K_MIN, eta: float = ETA_DEFAULT):
        """
        Initialize Phenomenology.
        
        Args:
            K: channels per horizon (D = 256 * K)
            eta: learning rate (default: aperture gap)
        """
        if K < 1:
            raise ValueError(f"K must be positive, got {K}")
        
        self.K = K
        self.D = 256 * K
        self.eta = eta
        
        # Precompute byte features for scoring
        self._byte_weights = np.array([mask_weight(b) for b in range(256)], dtype=np.float32)
        self._byte_charges = np.array([vertex_charge_for_byte(b) for b in range(256)], dtype=np.int32)
        self._byte_features = byte_feature_matrix(self.K)  # (256, K)
    
    def reshape(self, v: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Reshape external vector to horizon × channel tensor.
        
        Args:
            v: (D,) vector where D = 256 * K
            
        Returns:
            X: (256, K) tensor
        """
        if v.shape[0] != self.D:
            raise ValueError(f"Expected D={self.D}, got {v.shape[0]}")
        return v.reshape(256, self.K)
    
    def extract_local(self, X: NDArray[np.float32], h: int) -> NDArray[np.float32]:
        """
        Extract local activation at horizon index.
        
        Args:
            X: (256, K) tensor
            h: horizon index
            
        Returns:
            a: (K,) local activation
        """
        return X[h, :].copy()
    
    def update(
        self,
        state: PhenomenologyState,
        h_curr: int,
        a_curr: NDArray[np.float32],
        delta_mask: int,
    ) -> None:
        """
        Order-sensitive Hebbian update of M.
        
        Args:
            state: current phenomenology state (modified in place)
            h_curr: current horizon index
            a_curr: current local activation (K,)
            delta_mask: 12-bit mask for transition
        """
        if state.a_prev is None:
            # First step: just record, no update
            state.h_prev = h_curr
            state.a_prev = a_curr.copy()
            return
        
        # Compute direction factor (order-sensitive)
        gamma = direction_factor(state.h_prev, h_curr, delta_mask)
        
        # Hebbian increment: gamma * (current ⊙ previous)
        delta = gamma * (a_curr * state.a_prev)
        
        # Update M at current horizon position
        state.M[h_curr, :] += self.eta * delta
        
        # Numerical stability: bound M growth
        np.clip(state.M[h_curr, :], -M_CLIP, M_CLIP, out=state.M[h_curr, :])
        
        # Update previous state
        state.h_prev = h_curr
        state.a_prev = a_curr.copy()
    
    def score_bytes(
        self,
        state: PhenomenologyState,
        h_curr: int,
        a_curr: NDArray[np.float32],
        chi_curr: int,
    ) -> NDArray[np.float32]:
        """
        Score all 256 candidate bytes.
        
        Uses only kernel observables:
        - M signal (accumulated × current)
        - Mask weight (prefer lighter moves)
        - Vertex charge coherence
        
        Args:
            state: current phenomenology state
            h_curr: current horizon index
            a_curr: current local activation (K,)
            chi_curr: current vertex charge
            
        Returns:
            scores: (256,) score for each byte
        """
        # M signal: how well accumulated M aligns with current activation (per-byte features)
        x = state.M[h_curr, :] + a_curr                 # (K,)
        signal = self._byte_features @ x                # (256,) varies by byte
        
        # Weight penalty: prefer lighter masks (less disruption)
        weight_term = 1.0 - self._byte_weights
        
        # Vertex coherence: bonus for staying in same wedge
        wedge_match = (self._byte_charges == chi_curr).astype(np.float32)
        
        # Combine with fixed coefficients
        # (could be tuned, but keeping it simple and deterministic)
        scores = (
            0.5 * signal +
            0.3 * weight_term +
            0.2 * wedge_match
        )
        
        return scores
    
    def select_byte(
        self,
        state: PhenomenologyState,
        h_curr: int,
        a_curr: NDArray[np.float32],
        chi_curr: int,
        deterministic: bool = True,
    ) -> int:
        """
        Select next byte based on phenomenology.
        
        Args:
            state: current phenomenology state
            h_curr: current horizon index
            a_curr: current local activation (K,)
            chi_curr: current vertex charge
            deterministic: if True, use argmax; else sample from softmax
            
        Returns:
            selected byte ∈ {0..255}
        """
        scores = self.score_bytes(state, h_curr, a_curr, chi_curr)
        
        if deterministic:
            return int(np.argmax(scores))
        else:
            # Softmax sampling
            exp_scores = np.exp(scores - np.max(scores))  # stability
            probs = exp_scores / np.sum(exp_scores)
            return int(np.random.choice(256, p=probs))
    
    def step(
        self,
        state: PhenomenologyState,
        v_curr: NDArray[np.float32],
        h_curr: int,
        delta_mask: int,
        chi_curr: int,
        deterministic: bool = True,
    ) -> int:
        """
        Complete phenomenology step: update M and select byte.
        
        Args:
            state: phenomenology state (modified in place)
            v_curr: current external vector (D,)
            h_curr: current horizon index
            delta_mask: mask for transition (from previous step)
            chi_curr: current vertex charge
            deterministic: byte selection mode
            
        Returns:
            selected byte
        """
        # Reshape to horizon × channel
        X_curr = self.reshape(v_curr)
        
        # Extract local activation
        a_curr = self.extract_local(X_curr, h_curr)
        
        # Update M (order-sensitive Hebbian)
        self.update(state, h_curr, a_curr, delta_mask)
        
        # Select byte
        return self.select_byte(state, h_curr, a_curr, chi_curr, deterministic)