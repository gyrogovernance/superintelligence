"""
Inference Function (formerly Phenomenology operator).

Implements phase-aware inference with:
- Reshape: external vector → horizon × channel tensor
- Hebbian update: order-sensitive accumulation in M[h, p, :]
- Byte scoring: deterministic selection using kernel observables
- Spectral phase axis for frequency-aware memory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from src.agent.information import (
    ETA_DEFAULT,
    K_MIN,
    K_VALUES,
    M_CLIP,
)


@dataclass
class InferenceState:
    """
    Accumulated inference state with phase-aware memory.
    
    M: (256, 4, K) horizon × phase × channel field
    h_prev: previous horizon index
    p_prev: previous phase
    a_prev: previous local activation
    """
    M: NDArray[np.float32]
    h_prev: int = 0
    p_prev: int = 0
    a_prev: Optional[NDArray[np.float32]] = None
    
    @classmethod
    def create(cls, K: int) -> "InferenceState":
        """Initialise with zero M."""
        return cls(
            M=np.zeros((256, 4, K), dtype=np.float32),
            h_prev=0,
            p_prev=0,
            a_prev=None,
        )


class InferenceFunction:
    """
    Inference Function for Gyroscopic ASI.
    
    Maps external vectors + kernel state → byte selection,
    accumulating trajectory-dependent structure in M[h, p, :].
    
    The spectral phase axis (p ∈ {0,1,2,3}) captures position in
    byte-permutation cycles, enabling frequency-aware memory.
    """
    
    def __init__(self, K: int = K_MIN, eta: float = ETA_DEFAULT):
        """
        Initialise Inference Function.
        
        Args:
            K: channels per horizon (D = 256 * K)
            eta: learning rate (default: aperture gap)
        """
        if K not in K_VALUES:
            raise ValueError(f"K={K} is not supported. Supported K values: {K_VALUES}")
        
        self.K = K
        self.D = 256 * K
        self.eta = eta
        
        # Precomputed byte properties (will be loaded from kernel if available)
        self._byte_weights: NDArray[np.float32]
        self._byte_charges: NDArray[np.uint8]
        self._byte_features: NDArray[np.float32]
        self._gamma_table: NDArray[np.float32]
    
    def set_kernel_tables(
        self,
        byte_weight: NDArray[np.uint8],
        byte_charge: NDArray[np.uint8],
        byte_features: NDArray[np.float32],
        gamma_table: NDArray[np.float32],
    ) -> None:
        """Load precomputed tables from kernel's phenomenology."""
        self._byte_weights = byte_weight.astype(np.float32) / 12.0
        self._byte_charges = byte_charge
        self._byte_features = byte_features
        self._gamma_table = gamma_table
    
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
    
    def prepare_field(self, v: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Prepare field for inference (alias for reshape).
        
        Call this once per token to avoid redundant reshaping.
        """
        return self.reshape(v)
    
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
        state: InferenceState,
        h_curr: int,
        p_curr: int,
        a_curr: NDArray[np.float32],
        delta_mask: int,
        chi_prev: int,
        chi_curr: int,
    ) -> None:
        """
        Phase-aware Hebbian update of M.
        
        Args:
            state: current inference state (modified in place)
            h_curr: current horizon index
            p_curr: current phase (0-3)
            a_curr: current local activation (K,)
            delta_mask: 12-bit mask for transition
            chi_prev: previous vertex charge
            chi_curr: current vertex charge
        """
        if state.a_prev is None:
            state.h_prev = h_curr
            state.p_prev = p_curr
            state.a_prev = a_curr.copy()
            return
        
        # Compute direction factor using gamma table
        w = int(delta_mask).bit_count()
        
        # Assert bounds
        if w > 12:
            raise ValueError(f"delta_mask popcount out of range: {w} (mask={delta_mask:#x})")
        
        gamma = float(self._gamma_table[chi_prev, chi_curr, w])
        
        # Hebbian increment: gamma * (current ⊙ previous)
        delta = gamma * (a_curr * state.a_prev)
        
        # Update M at current horizon and phase position
        state.M[h_curr, p_curr, :] += self.eta * delta
        
        # Numerical stability: bound M growth
        np.clip(state.M[h_curr, p_curr, :], -M_CLIP, M_CLIP, out=state.M[h_curr, p_curr, :])
        
        # Update previous state
        state.h_prev = h_curr
        state.p_prev = p_curr
        state.a_prev = a_curr.copy()
    
    def score_bytes(
        self,
        state: InferenceState,
        h_curr: int,
        p_curr: int,
        a_curr: NDArray[np.float32],
        chi_curr: int,
    ) -> NDArray[np.float32]:
        """
        Score all 256 candidate bytes.
        
        Uses kernel observables:
        - M signal (accumulated × current)
        - Mask weight (prefer lighter moves)
        - Vertex charge coherence
        
        Args:
            state: current inference state
            h_curr: current horizon index
            p_curr: current phase (0-3)
            a_curr: current local activation (K,)
            chi_curr: current vertex charge
            
        Returns:
            scores: (256,) score for each byte
        """
        # M signal: how well accumulated M aligns with current activation
        x = state.M[h_curr, p_curr, :] + a_curr
        
        signal = self._byte_features @ x
        
        # Weight penalty: prefer lighter masks
        weight_term = 1.0 - self._byte_weights
        
        # Vertex coherence: bonus for staying in same wedge
        wedge_match = (self._byte_charges == chi_curr).astype(np.float32)
        
        # Combine with fixed coefficients
        scores = (
            0.5 * signal +
            0.3 * weight_term +
            0.2 * wedge_match
        )
        
        return scores
    
    def select_byte(
        self,
        scores: NDArray[np.float32],
        deterministic: bool = True,
        allowed_max_byte: int = 255,
        allowed_mask: Optional[NDArray[np.bool_]] = None,
    ) -> int:
        """
        Select byte from scores.
        
        Args:
            scores: (256,) score for each byte
            deterministic: if True, use argmax; else sample from softmax
            allowed_max_byte: maximum allowed byte value (prefix gating)
            allowed_mask: optional boolean mask shape (256,) for non-contiguous gating
            
        Returns:
            selected byte ∈ {0..allowed_max_byte} ∩ allowed_mask
        """
        limit = allowed_max_byte + 1
        s = scores[:limit]
        
        if allowed_mask is not None:
            allowed_mask = np.asarray(allowed_mask, dtype=np.bool_)
            if allowed_mask.shape != (256,):
                raise ValueError(f"allowed_mask must be shape (256,), got {allowed_mask.shape}")
            m = allowed_mask[:limit]
            if not m.any():
                raise ValueError("No allowed bytes under current gating")
            
            if deterministic:
                masked = np.where(m, s, -np.inf)
                return int(np.argmax(masked))
            else:
                idx = np.flatnonzero(m)
                s2 = s[idx]
                exp_s = np.exp(s2 - np.max(s2))
                probs = exp_s / np.sum(exp_s)
                return int(idx[np.random.choice(len(idx), p=probs)])
        
        # No mask: only prefix gating
        if deterministic:
            return int(np.argmax(s))
        else:
            exp_s = np.exp(s - np.max(s))
            probs = exp_s / np.sum(exp_s)
            return int(np.random.choice(limit, p=probs))
    
    def step_with_field(
        self,
        state: InferenceState,
        X_curr: NDArray[np.float32],
        h_curr: int,
        p_curr: int,
        delta_mask: int,
        chi_prev: int,
        chi_curr: int,
        deterministic: bool = True,
        allowed_max_byte: int = 255,
        allowed_mask: Optional[NDArray[np.bool_]] = None,
    ) -> int:
        """
        Complete inference step with pre-reshaped field.
        
        Args:
            state: inference state (modified in place)
            X_curr: current field (256, K) - already reshaped
            h_curr: current horizon index
            p_curr: current phase (0-3)
            delta_mask: mask for transition (from previous step)
            chi_prev: previous vertex charge
            chi_curr: current vertex charge
            deterministic: byte selection mode
            allowed_max_byte: maximum allowed byte (prefix gating)
            allowed_mask: optional boolean mask for non-contiguous gating
            
        Returns:
            selected byte
        """
        # Guard: ensure all required kernel tables are loaded
        required_tables = ["_gamma_table", "_byte_features", "_byte_weights", "_byte_charges"]
        missing_tables = [table for table in required_tables if not hasattr(self, table)]
        if missing_tables:
            raise RuntimeError(
                f"InferenceFunction tables not set. Missing: {missing_tables}. "
                "Call set_kernel_tables() first."
            )
        
        # Bounds validation to prevent silent out-of-range indexing
        if not (0 <= h_curr <= 255):
            raise ValueError(f"horizon index h_curr={h_curr} out of range [0, 255]")
        if not (0 <= p_curr <= 3):
            raise ValueError(f"phase p_curr={p_curr} out of range [0, 3]")
        if not (0 <= allowed_max_byte <= 255):
            raise ValueError(f"allowed_max_byte={allowed_max_byte} out of range [0, 255]")
        
        # Extract local activation
        a_curr = self.extract_local(X_curr, h_curr)
        
        # Update M (phase-aware Hebbian)
        self.update(state, h_curr, p_curr, a_curr, delta_mask, chi_prev, chi_curr)
        
        # Score bytes
        scores = self.score_bytes(state, h_curr, p_curr, a_curr, chi_curr)
        
        # Select byte (with optional mask)
        return self.select_byte(scores, deterministic, allowed_max_byte, allowed_mask)