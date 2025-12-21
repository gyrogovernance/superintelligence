"""
GyroSI Core Engine: Walking Intelligence on a 48-bit Geometric Manifold

This module implements the core walking mechanics of GyroSI, where intelligence
emerges through recursive navigation of a finite state space using gyrogroup
operations and monodromic folding.

The system operates on five canonical maps derived from the Common Governance Model:
- Theta: Angular divergence from archetypal state (CS)
- Ontology: Complete discovered manifold (UNA) 
- Epistemology: State transition table (BU-Egress)
- Phenomenology: Canonical orbit representatives (ONA)
- Orbit Sizes: Cardinality structure (BU-Ingress)
"""

import numpy as np
import pickle
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import gcd


# Boundary transformations between byte-space and intron-space
def byte_to_intron(byte_value: int) -> int:
    """Transform external byte to internal intron via ψ boundary (XOR 0xAA)."""
    return (byte_value & 0xFF) ^ 0xAA


def intron_to_byte(intron_value: int) -> int:
    """Transform internal intron to external byte via ψ⁻¹ boundary (XOR 0xAA)."""
    return (intron_value & 0xFF) ^ 0xAA


def token_to_introns(token_id: int) -> List[int]:
    """
    Convert Harmony token ID to sequence of introns via big-endian bytes.
    
    Args:
        token_id: Integer token identifier
        
    Returns:
        List of introns (bytes transformed through ψ boundary)
        
    Raises:
        ValueError: If token_id is negative
    """
    if token_id < 0:
        raise ValueError("Negative token id")
    byte_length = (token_id.bit_length() + 7) // 8 or 1
    bytes_sequence = token_id.to_bytes(byte_length, "big")
    return [byte_to_intron(b) for b in bytes_sequence]


class GyroEngine:
    """
    Core walking engine implementing intelligence as navigation on a geometric manifold.
    
    The engine operates through two primary phases:
    - BU-Egress: Absorbing input and updating internal state (stance phase)
    - BU-Ingress: Generating output through phase-propagating emission (swing phase)
    
    Memory Structure:
    - Active Memory: 6-bit ctx6 per slab (CS - immediate dynamic context)
    - Session Memory: 48-bit state + 8-bit walk_phase (UNA/ONA - trajectory/momentum)
    - Passive Memory: (rep, slab, ctx6) → List[int] token IDs (BU - compact holographic surfaces)
    - Adjacency: (rep, slab, ctx6, prev_token) → List[int] successors (sequence linking)
    
    Lossless trajectory recorded in Atlas Ledger; adjacency enables ordered reproduction.
    Walking continues until natural amplitude decay (balance achieved).
    """

    def __init__(
        self,
        atlas_paths: Dict[str, str],
        store_paths: Optional[Dict[str, str]] = None,
        runtime: Optional[Dict[str, str]] = None,
        version_info: Optional[Dict[str, str]] = None,
        vocab_size: int = 201_088,
        enable_core_gate: bool = False,
    ):
        """
        Initialize the walking engine with the five canonical maps.
        
        Args:
            atlas_paths: Paths to the five canonical map files
            store_paths: Optional paths for persistent storage
            runtime: Runtime configuration parameters
            version_info: Version information
            vocab_size: Maximum vocabulary size
            enable_core_gate: Enable refractory gating mechanisms
        """
        # Load the five canonical maps
        self.theta = np.load(atlas_paths["theta"], mmap_mode="r")
        self.ontology_keys = np.load(atlas_paths["ontology_keys"], mmap_mode="r")
        self.epistemology = np.load(atlas_paths["epistemology"], mmap_mode="r")
        self.phenomenology_map = np.load(atlas_paths["phenomenology_map"], mmap_mode="r")
        self.orbit_sizes = np.load(atlas_paths["orbit_sizes"], mmap_mode="r")

        # Build reverse index for state lookup
        self.state_to_index: Dict[int, int] = {
            int(state): int(index) for index, state in enumerate(self.ontology_keys)
        }

        # Extract canonical orbit representatives
        representatives = np.unique(self.phenomenology_map)
        self.orbit_representatives: List[int] = [int(rep) for rep in representatives]

        # Walking memory structures
        self.orbit_phase: Dict[int, int] = {}  # Per-orbit phase accumulator
        self.slab_channels: Dict[Tuple[int, int], Dict[int, List[int]]] = {}  # Slab-specific token storage
        
        # Minimal adjacency for sequence linking
        self.adjacency: Dict[Tuple[int, int, int, int], List[int]] = {}  # (rep, slab_idx, ctx6, prev_token) → successors
        self.last_token_in_ctx: Dict[Tuple[int, int, int], int] = {}  # (rep, slab_idx, ctx6) → last token
        self.last_emitted_token_in_ctx: Dict[Tuple[int, int, int], int] = {}  # (rep, slab_idx, ctx6) → last emitted token

        # Emission hygiene for path-native walking
        self.emission_gate: Dict[int, int] = {}
        self.last_emitted_token: Dict[int, int] = {}
        self.last_emission_tick: Dict[int, int] = {}

        # Configuration
        self.store_paths = store_paths or {}
        self.runtime = runtime or {}
        self.version_info = version_info or {}
        self.vocab_size = vocab_size
        self.vocab_max = vocab_size

        # Walking physics switches
        self.enable_core_gate = enable_core_gate

        # Override from runtime if specified
        if "enable_core_gate" in (self.runtime or {}):
            self.enable_core_gate = bool(self.runtime["enable_core_gate"])

        # Persistence control
        self._token_counter = 0
        self._last_save_time = time.time()
        self._save_interval_tokens = 100
        self._save_interval_seconds = 30.0
        self._pending_changes = False
        self._max_bucket_size = 64

        # Thread safety
        self._lock = threading.RLock()

        # Find archetypal starting state (minimum theta)
        self.start_index: int = int(np.argmin(self.theta))
        
        # Atlas Ledger for lossless trajectory recording
        self._atlas_ledger_enabled = "atlas_ledger" in (self.store_paths or {})

        # Initialize storage and load existing data
        self._ensure_storage_files()
        self._load_learned_data()

    # State space navigation methods
    def get_theta(self, index: int) -> float:
        """Get angular divergence from archetype for given state index."""
        return float(self.theta[index])

    def get_orbit_representative(self, index: int) -> int:
        """Get canonical orbit representative index for given state."""
        return int(self.phenomenology_map[index])

    def apply_intron_to_index(self, index: int, intron: int) -> int:
        """
        Apply intron transformation using holographic physics.
        
        Each intron acts as a quantum of action, broadcasting holographically
        across all 48 bits of the state tensor through EXON family operations.
        """
        if not (0 <= intron <= 255):
            raise ValueError("intron out of range")
        
        from baby.kernel.governance import apply_gyration_and_transform
        
        state_before = int(self.ontology_keys[index])
        # Physics transform: holographic broadcast + EXON family action
        state_after = apply_gyration_and_transform(state_before, intron)
        new_index = self.state_to_index.get(state_after)
        
        if new_index is None:
            # Fallback to atlas transition if needed (shouldn't happen if atlas was built from the same physics)
            new_index = int(self.epistemology[index, intron])
            
        return new_index

    # Monodromic fold operations
    @staticmethod
    def monodromic_fold(accumulator: int, intron: int) -> int:
        """
        Core monodromic fold operation: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
        
        This non-associative operation preserves path memory and enables
        time as the sequential ordering of recursive operations.
        """
        a = accumulator & 0xFF
        b = intron & 0xFF
        return (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF

    def fold_sequence(self, introns: List[int], initial: int = 0) -> Tuple[int, int]:
        """
        Fold sequence of introns returning phase and amplitude.
        
        Args:
            introns: Sequence of intron values
            initial: Initial accumulator value
            
        Returns:
            Tuple of (final_phase, final_amplitude)
        """
        accumulator = initial & 0xFF
        amplitude = 0
        for intron in introns:
            accumulator, amplitude = self._fold_with_amplitude(accumulator, intron)
        return accumulator, amplitude

    @staticmethod
    def _fold_with_amplitude(a: int, b: int) -> Tuple[int, int]:
        """Fold operation returning both result and amplitude (bit count)."""
        a &= 0xFF
        b &= 0xFF
        result = (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF
        amplitude = bin(result).count('1')
        return result, amplitude

    # State phase computation
    def compute_state_phase(self, state_int: int) -> Tuple[int, int]:
        """
        Project 48-bit state into 8-bit phase and velocity through ψ boundary.
        
        The state's 6 bytes are folded sequentially after ψ transformation,
        with velocity computed as the delta between successive folds.
        """
        byte_sequence = int(state_int).to_bytes(6, "big")
        phase = 0
        previous_phase = 0
        velocity = 0
        
        for byte_value in byte_sequence:
            previous_phase = phase
            phase, _ = self._fold_with_amplitude(phase, byte_value ^ 0xAA)
            velocity, _ = self._fold_with_amplitude(velocity, phase ^ previous_phase)
            
        return phase, velocity

    def compute_token_phase(self, token_id: int) -> Tuple[int, int]:
        """Compute phase and amplitude for token through intron folding."""
        return self.fold_sequence(token_to_introns(token_id), 0)

    # Six degrees of freedom computation
    def compute_six_dof(self, state_int: int) -> Tuple[int, int, int, int, int, int]:
        """
        Extract six degrees of freedom from state as rotational and translational phases.
        
        Rotational (rX, rY, rZ): Per-row parity across both frames
        Translational (tX, tY, tZ): Frame-difference parity per row
        
        Returns:
            Tuple of (rX, rY, rZ, tX, tY, tZ) as 8-bit phases
        """
        # Rotational freedoms (both frames combined)
        rX = self._compute_row_parity(state_int, row=0, frame=None)
        rY = self._compute_row_parity(state_int, row=1, frame=None)
        rZ = self._compute_row_parity(state_int, row=2, frame=None)

        # Translational freedoms (frame differences)
        f0X = self._compute_row_parity(state_int, row=0, frame=0)
        f1X = self._compute_row_parity(state_int, row=0, frame=1)
        f0Y = self._compute_row_parity(state_int, row=1, frame=0)
        f1Y = self._compute_row_parity(state_int, row=1, frame=1)
        f0Z = self._compute_row_parity(state_int, row=2, frame=0)
        f1Z = self._compute_row_parity(state_int, row=2, frame=1)

        tX, _ = self._fold_with_amplitude(f0X, f1X)
        tY, _ = self._fold_with_amplitude(f0Y, f1Y)
        tZ, _ = self._fold_with_amplitude(f0Z, f1Z)

        return rX, rY, rZ, tX, tY, tZ

    def _compute_row_parity(self, state_int: int, row: int, frame: Optional[int] = None) -> int:
        """Compute parity fold for tensor row across specified frames."""
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        
        accumulator = 0
        for layer in range(FROZEN_CHANNELS.NUM_LAYERS):
            frame_range = range(FROZEN_CHANNELS.NUM_FRAMES) if frame is None else [frame]
            for fr in frame_range:
                for col in range(FROZEN_CHANNELS.NUM_COLS):
                    bit_index = FROZEN_CHANNELS.get_bit_index(layer, fr, row, col)
                    bit = (state_int >> bit_index) & 1
                    accumulator, _ = self._fold_with_amplitude(accumulator, (bit & 1) * 0x01)
        return accumulator

    # Slab operations for the walking model
    def extract_slab_byte(self, state_int: int, slab_index: int) -> int:
        """
        Extract 6-bit slab from state and compress to byte with ψ transformation.
        
        Each slab represents a body segment in the walking model:
        - Slabs 0,7: Boundaries (head/foot orientation)
        - Slabs 1-6: Six degrees of freedom for movement
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        
        bit_indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_index)
        byte_value = 0
        
        for position, bit_index in enumerate(bit_indices):
            byte_value |= ((state_int >> bit_index) & 1) << position
            
        return (byte_value ^ 0xAA) & 0xFF

    def _slab_ctx6(self, state_int: int, slab_index: int) -> int:
        """
        Extract 6-bit dynamic context from slab, ignoring L0 anchors.
        
        This provides the minimal context for active memory addressing
        that aligns with intron anatomy and reduces key space from 256 to 64.
        """
        return self.extract_slab_byte(state_int, slab_index) & 0x3F

    def _adj_push(self, rep: int, slab_idx: int, ctx6: int, prev_tok: int, tok: int, cap: int = 16) -> None:
        """Add adjacency edge for sequence linking."""
        lst = self.adjacency.setdefault((rep, slab_idx, ctx6, prev_tok), [])
        if tok in lst:
            lst.remove(tok)
        lst.append(tok)
        if len(lst) > cap:
            lst.pop(0)

    def compute_sector_signature(self, state_int: int) -> int:
        """
        Compute 8-bit toroidal sector signature from state slab parities.
        
        Each bit represents the parity of one slab, creating an 8-bit
        signature that determines active walking regions.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS

        sector_bits = 0
        for slab_index in range(FROZEN_CHANNELS.NUM_SLABS):
            parity = 0
            for bit_index in FROZEN_CHANNELS.get_slab_bit_indices(slab_index):
                if (state_int >> bit_index) & 1:
                    parity ^= 1
            if parity:
                sector_bits |= (1 << slab_index)

        return sector_bits

    # Temporal coupling for endogenous variation
    def compute_temporal_tick(self) -> int:
        """
        Generate endogenous temporal phase from monotonic clock via ψ boundary.
        
        This provides lawful temporal variation without randomness,
        coupling the walking rhythm to physical time.
        """
        if self.runtime.get("reproducible_ticks", False):
            return 0  # Deterministic mode for replay
        
        nanoseconds = time.time_ns()
        byte_sequence = nanoseconds.to_bytes(8, "big")[-6:]
        accumulator = 0
        for byte_value in byte_sequence:
            accumulator, _ = self._fold_with_amplitude(accumulator, byte_value ^ 0xAA)
        return accumulator

    def apply_temporal_twist(self, phase: int) -> int:
        """Apply helical time progression to phase for endogenous variation."""
        tick = self.compute_temporal_tick()
        twisted_phase, _ = self._fold_with_amplitude(phase, tick)
        return twisted_phase

    # Walking helper methods
    def compute_alignment_amplitude(self, state_int: int, representative_index: int, walk_phase: int = 0) -> int:
        """
        Compute amplitude of alignment between current state, walk momentum, and orbit memory.
        
        Zero amplitude indicates natural stopping condition (balance achieved).
        """
        state_phase, _ = self.compute_state_phase(state_int)
        orbit_phase = self.orbit_phase.get(representative_index, 0)
        
        # Combine live state with walk momentum before comparing to learned orbit direction
        live_phase, _ = self._fold_with_amplitude(state_phase, walk_phase)
        
        _, amplitude = self._fold_with_amplitude(live_phase, orbit_phase)
        return amplitude

    def compute_coprime_stride(self, seed: int, modulus: int) -> int:
        """Generate coprime stride for ring walking using fold-compatible arithmetic."""
        if modulus <= 1:
            return 1
        stride = (1 + (seed % modulus)) or 1
        
        # Ensure coprimality without thresholds
        while gcd(stride, modulus) != 1:
            stride = (stride + 1) % modulus
            if stride == 0:
                stride = 1
        return stride

    # BU-Egress: Learning and state absorption
    def learn_from_user_token(self, state: int, token_id: int) -> int:
        """
        BU-Egress: Absorb user token into memory and advance state (stance phase).
        
        This implements the stance phase of walking where external input
        is absorbed and integrated into the system's memory structure.
        """
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")

        index = self.state_to_index[state]
        representative = self.get_orbit_representative(index)
        
        # Compute token phase through intron folding
        introns = token_to_introns(token_id)
        token_phase, _ = self.fold_sequence(introns, 0)

        # Update orbit phase memory
        current_phase = self.orbit_phase.get(representative, 0)
        new_phase, _ = self._fold_with_amplitude(current_phase, token_phase)

        state_int = int(self.ontology_keys[index])

        with self._lock:
            # Store tokens in slab-specific channels using 6-bit dynamic contexts
            for slab_index in range(8):
                ctx6 = self._slab_ctx6(state_int, slab_index)
                channel_key = (representative, slab_index)
                slab_channel = self.slab_channels.setdefault(channel_key, {})
                bucket = slab_channel.setdefault(ctx6, [])
                
                # Add adjacency for sequence linking
                key_ctx = (representative, slab_index, ctx6)
                prev_tok = self.last_token_in_ctx.get(key_ctx)
                if prev_tok is not None and prev_tok != token_id:
                    self._adj_push(representative, slab_index, ctx6, prev_tok, token_id)
                self.last_token_in_ctx[key_ctx] = token_id
                
                # Store token with single FIFO capacity check
                if token_id not in bucket:
                    if len(bucket) >= self._max_bucket_size:
                        bucket.pop(0)
                    bucket.append(token_id)

            # Update orbit phase memory
            self.orbit_phase[representative] = new_phase
            self._pending_changes = True
            self._token_counter += 1

        # Advance state through intron sequence
        new_index = index
        for intron in introns:
            # Log egress event before applying intron
            self._log_event(0x01, int(self.ontology_keys[new_index]), intron)
            new_index = self.apply_intron_to_index(new_index, intron)

        # Store content in final state's representation as well
        final_representative = self.get_orbit_representative(new_index)
        final_state_int = int(self.ontology_keys[new_index])
        
        for slab_index in range(8):
            ctx6 = self._slab_ctx6(final_state_int, slab_index)
            channel_key = (final_representative, slab_index)
            slab_channel = self.slab_channels.setdefault(channel_key, {})
            bucket = slab_channel.setdefault(ctx6, [])
            
            # Add adjacency for sequence linking
            key_ctx = (final_representative, slab_index, ctx6)
            prev_tok = self.last_token_in_ctx.get(key_ctx)
            if prev_tok is not None and prev_tok != token_id:
                self._adj_push(final_representative, slab_index, ctx6, prev_tok, token_id)
            self.last_token_in_ctx[key_ctx] = token_id
            
            # Store token with single FIFO capacity check
            if token_id not in bucket:
                if len(bucket) >= self._max_bucket_size:
                    bucket.pop(0)
                bucket.append(token_id)

        self._maybe_save_learned_data()
        return int(self.ontology_keys[new_index])

    def transit_on_assistant_token(self, state: int, token_id: int) -> int:
        """Transit state on assistant token without learning."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        
        index = self.state_to_index[state]
        introns = token_to_introns(token_id)
        
        new_index = index
        for intron in introns:
            new_index = self.apply_intron_to_index(new_index, intron)
            
        return int(self.ontology_keys[new_index])

    def transit_on_control_token(self, state: int, token_id: int) -> int:
        """Control tokens do not affect state."""
        return state

    # BU-Ingress: Phase-propagating emission (swing phase)
    def emit_next_token(
        self,
        index: int,
        session_omega: Optional[Dict[Tuple[int,int], int]] = None,
        session_bucket_key: Optional[Dict[Tuple[int,int], int]] = None,
        session_bucket_position: Optional[Dict[Tuple[int,int], Dict[int, int]]] = None,
        session_monodromy: Optional[Dict[Tuple[int,int], int]] = None,
        recent_egress_phases: Optional[List[int]] = None,
        session_slab_cursor: Optional[Dict[int, int]] = None,
        session_walk_phase: int = 0,  # NEW
        session_prev_by_ctx: Optional[Dict[Tuple[int,int,int], int]] = None,  # NEW: Context-scoped predecessor
    ) -> Optional[Tuple[int, int, Dict[Tuple[int,int], int], Dict[Tuple[int,int], int], 
                       Dict[Tuple[int,int], Dict[int, int]], Dict[Tuple[int,int], int], Dict[int,int]]]:
        """
        BU-Ingress: Generate next token through phase-propagating emission (swing phase).
        
        This implements the swing phase of walking where forward momentum
        is generated based on current state and learned patterns.
        """
        representative = self.get_orbit_representative(index)
        state_int = int(self.ontology_keys[index])

        # Initialize session state if not provided
        omega = session_omega or {}
        bucket_key = session_bucket_key or {}
        bucket_position = session_bucket_position or {}
        monodromy = session_monodromy or {}
        slab_cursor = session_slab_cursor or {}

        # Natural stopping including walk momentum
        if self.compute_alignment_amplitude(state_int, representative, session_walk_phase) == 0:
            return None

        # Determine active slabs from sector signature
        sector_bits = self.compute_sector_signature(state_int)
        # Exclude boundary slabs (0 and 7) from generation - use dynamic slabs (1..6)
        active_slabs = [s for s in range(1, 7) if (sector_bits >> s) & 1]
        if not active_slabs:
            active_slabs = list(range(1, 7))

        # Round-robin slab traversal
        start_position = slab_cursor.get(representative, 0) % len(active_slabs)
        slab_order = [
            active_slabs[(start_position + hop) % len(active_slabs)] 
            for hop in range(len(active_slabs))
        ]

        # Compute live phases
        state_phase, state_velocity = self.compute_state_phase(state_int)
        temporal_tick = self.compute_temporal_tick()

        # Walk through slabs to find viable token
        for slab_index in slab_order:
            channel_key = (representative, slab_index)
            slab_channel = self.slab_channels.get(channel_key, {})
            if not slab_channel:
                continue

            context_keys = sorted(slab_channel.keys())
            num_contexts = len(context_keys)
            if num_contexts == 0:
                continue

            # Context addressing using 6-bit dynamic context
            ctx6 = self._slab_ctx6(state_int, slab_index)

            # Initialize rotor if first time
            if channel_key not in bucket_key:
                bucket_key[channel_key] = context_keys[ctx6 % num_contexts]
                if channel_key not in bucket_position:
                    bucket_position[channel_key] = {}

            # Compute ring stride
            import bisect
            base_value = bucket_key[channel_key]
            base_index = bisect.bisect_left(context_keys, base_value) % num_contexts

            omega_value = omega.get(channel_key, 0)
            stride_seed, _ = self._fold_with_amplitude(state_phase, omega_value ^ temporal_tick)
            stride_seed, _ = self._fold_with_amplitude(stride_seed, session_walk_phase)  # NEW
            stride = self.compute_coprime_stride(stride_seed, num_contexts)

            # Walk the context ring
            for step_count in range(num_contexts):
                current_index = (base_index + step_count * stride) % num_contexts
                current_key = context_keys[current_index]
                bucket = slab_channel.get(current_key, [])
                if not bucket:
                    continue

                # Intra-bucket rotor
                mono_value = monodromy.get(channel_key, 0)
                inner_seed, _ = self._fold_with_amplitude(state_velocity, mono_value ^ temporal_tick)
                inner_seed, _ = self._fold_with_amplitude(inner_seed, session_walk_phase)  # NEW
                bucket_length = len(bucket)
                inner_stride = self.compute_coprime_stride(inner_seed, bucket_length)

                position_map = bucket_position.setdefault(channel_key, {})
                base_position = position_map.get(current_key, 0)

                # Get successors for adjacency-based rotor positioning using context-scoped tracking
                prev_tok = (session_prev_by_ctx.get((representative, slab_index, ctx6), -1)
                           if session_prev_by_ctx is not None
                           else self.last_emitted_token_in_ctx.get((representative, slab_index, ctx6), -1))
                succ = self.adjacency.get((representative, slab_index, ctx6, prev_tok), [])
                
                # Bias rotor base position if successors exist
                succ_hit = False
                if succ:
                    for s in reversed(succ):  # prefer most recent
                        if s in bucket:
                            base_position = bucket.index(s)
                            succ_hit = True
                            break
                if not succ_hit:
                    base_position = position_map.get(current_key, 0)

                # Try tokens in bucket
                for attempt in range(bucket_length):
                    position = (base_position + attempt * inner_stride) % bucket_length
                    candidate_token = bucket[position]
                    candidate_phase, _ = self.fold_sequence(token_to_introns(candidate_token), 0)

                    # Apply emission gates
                    gate_accumulator = self.emission_gate.get(representative, 0)
                    gated_value, _ = self._fold_with_amplitude(gate_accumulator, candidate_phase)

                    # Core gate filtering
                    passes_core_gate = True
                    if self.enable_core_gate:
                        last_token = self.last_emitted_token.get(representative, -1)
                        last_tick = self.last_emission_tick.get(representative, temporal_tick)
                        tick_delta = (temporal_tick - last_tick) & 0xFF
                        
                        # Egress mask: avoid recently emitted phases
                        if recent_egress_phases:
                            token_phase, _ = self.compute_token_phase(candidate_token)
                            if token_phase in recent_egress_phases:
                                passes_core_gate = False
                        
                        # Refractory gate: suppress immediate repeats
                        if (passes_core_gate and candidate_token == last_token and 
                            self._fold_with_amplitude(tick_delta, candidate_phase)[0] == 0):
                            passes_core_gate = False

                    # Chain extension check - does this chain extend the current walk trajectory?
                    extended_phase = self.monodromic_fold(session_walk_phase, candidate_phase)
                    extends_chain = extended_phase != 0
                    
                    # Note: Resonance and FG checks removed as hard constraints
                    # They can be used as bias in future enhancements if needed

                    # Amplitude-preserving step constraint
                    pred_index = index
                    for intron in token_to_introns(candidate_token):
                        pred_index = self.apply_intron_to_index(pred_index, intron)
                    pred_state_int = int(self.ontology_keys[pred_index])
                    pred_walk_phase = self.monodromic_fold(session_walk_phase, candidate_phase)
                    pred_amp = self.compute_alignment_amplitude(pred_state_int, representative, pred_walk_phase)
                    # Don't choke on amplitude when memory is immature
                    orbit_phase_now = self.orbit_phase.get(representative, 0)
                    amplitude_ok = True if orbit_phase_now == 0 else (pred_amp != 0)

                    if gated_value != 0 and passes_core_gate and amplitude_ok and extends_chain:
                        # Token selected - update traces and return
                        self.emission_gate[representative] = gated_value
                        self.last_emitted_token[representative] = candidate_token
                        self.last_emitted_token_in_ctx[(representative, slab_index, ctx6)] = candidate_token
                        # Update context-scoped tracking if provided
                        if session_prev_by_ctx is not None:
                            session_prev_by_ctx[(representative, slab_index, ctx6)] = candidate_token
                        self.last_emission_tick[representative] = temporal_tick

                        # Update path memory
                        omega[channel_key], _ = self._fold_with_amplitude(
                            (omega.get(channel_key, 0) + 1) & 0xFF, candidate_phase
                        )
                        monodromy[channel_key], _ = self._fold_with_amplitude(
                            mono_value, candidate_phase
                        )

                        # Advance rotor positions
                        bucket_key[channel_key] = context_keys[(current_index + stride) % num_contexts]
                        position_map[current_key] = (position + 1) % bucket_length

                        # Apply token introns to advance state
                        new_index = index
                        for intron in token_to_introns(candidate_token):
                            # Log emission event before applying intron
                            self._log_event(0x02, int(self.ontology_keys[new_index]), intron)
                            new_index = self.apply_intron_to_index(new_index, intron)

                        # Advance slab cursor only if no adjacency hit (maintain slab continuity)
                        if not succ_hit:
                            slab_cursor[representative] = ((slab_cursor.get(representative, 0) + 1) % 
                                                         max(1, len(active_slabs)))

                        return (candidate_token, new_index, omega, bucket_key, 
                               bucket_position, monodromy, slab_cursor)

        # No coherent continuation found
        return None

    # Harmony integration
    def is_harmony_control_token(self, token_id: int) -> bool:
        """Check if token is a Harmony control token."""
        from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS
        return token_id in ALL_CONTROL_TOKENS

    def should_learn_from_token(self, token_id: int, role: str) -> bool:
        """Determine if token should trigger learning based on role and type."""
        return (role == "user" and 
                not self.is_harmony_control_token(token_id) and 
                0 <= token_id < self.vocab_max)

    # Interface compatibility methods
    def start_state(self) -> int:
        """Get the archetypal starting state."""
        return int(self.ontology_keys[self.start_index])

    def get_state_theta(self, state: int) -> float:
        """Get theta (angular divergence) for state."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        index = self.state_to_index[state]
        return self.get_theta(index)

    def get_state_orbit_representative(self, state: int) -> int:
        """Get orbit representative state for given state."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        index = self.state_to_index[state]
        rep_index = self.get_orbit_representative(index)
        return int(self.ontology_keys[rep_index])

    def get_state_orbit_size(self, state: int) -> int:
        """Get orbit size for given state."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        index = self.state_to_index[state]
        return int(self.orbit_sizes[index])

    def apply_intron_to_state(self, state: int, intron: int) -> int:
        """Apply intron to state and return new state."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        index = self.state_to_index[state]
        new_index = self.apply_intron_to_index(index, intron)
        return int(self.ontology_keys[new_index])

    def compute_micro_path(self, start_state: int, introns: List[int]) -> List[int]:
        """Compute path through state space following intron sequence."""
        if start_state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{start_state:012X}")
        
        index = self.state_to_index[start_state]
        path = [start_state]
        current_index = index
        
        for intron in introns:
            current_index = self.apply_intron_to_index(current_index, intron)
            path.append(int(self.ontology_keys[current_index]))
            
        return path

    # Wrapper methods for backward compatibility
    def evolve_on_user(self, state: int, token_id: int) -> int:
        return self.learn_from_user_token(state, token_id)

    def evolve_on_assistant(self, state: int, token_id: int) -> int:
        return self.transit_on_assistant_token(state, token_id)

    def emit_next_from_state(self, state_int: int, **kwargs) -> Optional[Tuple[int, int, Dict[Tuple[int,int], int], Dict[Tuple[int,int], int], Dict[Tuple[int,int], Dict[int, int]], Dict[Tuple[int,int], int], Dict[int,int]]]:
        if state_int not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_int:012X}")
        index = self.state_to_index[state_int]
        result = self.emit_next_token(index, **kwargs)
        if result is None:
            return None
        token, new_index, omega, bucket_key, bucket_position, monodromy, slab_cursor = result
        return (token, int(self.ontology_keys[new_index]), omega, bucket_key, bucket_position, monodromy, slab_cursor)

    def next_token_aligned(self, state: int) -> Optional[int]:
        result = self.emit_next_from_state(state)
        return None if result is None else result[0]

    def next_token(self, state: int) -> Optional[int]:
        result = self.emit_next_from_state(state)
        return None if result is None else result[0]

    # Persistence methods
    def _ensure_storage_files(self):
        """Ensure storage directories and files exist."""
        if not self.store_paths:
            return
        for path_string in self.store_paths.values():
            path = Path(path_string)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()

    def _load_learned_data(self):
        """Load persistent learning data from disk."""
        if not self.store_paths:
            return

        with self._lock:
            # Load orbit phase data
            phase_path = self.store_paths.get("rep_phase")
            if phase_path and os.path.exists(phase_path) and os.path.getsize(phase_path) > 0:
                try:
                    with open(phase_path, "rb") as f:
                        self.orbit_phase = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    self.orbit_phase = {}

            # Load channel data with format migration
            channel_path = self.store_paths.get("rep_channel")
            if channel_path and os.path.exists(channel_path) and os.path.getsize(channel_path) > 0:
                try:
                    with open(channel_path, "rb") as f:
                        loaded_channels = pickle.load(f)

                    # Check for old format and migrate if needed
                    if loaded_channels and isinstance(list(loaded_channels.keys())[0], int):
                        # Old format detected - start fresh with new slab-based format
                        self.slab_channels = {}
                    else:
                        # New format - use as is
                        self.slab_channels = loaded_channels

                except (pickle.UnpicklingError, EOFError):
                    self.slab_channels = {}

    def _save_learned_data(self):
        """Save learning data to persistent storage."""
        if not self.store_paths:
            return

        with self._lock:
            if not self._pending_changes:
                return

            phase_path = self.store_paths.get("rep_phase")
            if phase_path:
                with open(phase_path, "wb") as f:
                    pickle.dump(self.orbit_phase, f)

            channel_path = self.store_paths.get("rep_channel")
            if channel_path:
                with open(channel_path, "wb") as f:
                    pickle.dump(self.slab_channels, f)

            self._pending_changes = False
            self._last_save_time = time.time()
            self._token_counter = 0

    def _maybe_save_learned_data(self):
        """Conditionally save data based on time and token thresholds."""
        current_time = time.time()
        time_elapsed = current_time - self._last_save_time

        should_save = (
            self._pending_changes and
            (self._token_counter >= self._save_interval_tokens or
             time_elapsed >= self._save_interval_seconds)
        )

        if should_save:
            self._save_learned_data()

    def _log_event(self, event_code: int, state_int: int, intron: Optional[int] = None) -> None:
        """Log intron events to Atlas Ledger for lossless trajectory recording."""
        if not self._atlas_ledger_enabled:
            return
        
        ledger_path = self.store_paths.get("atlas_ledger")
        if not ledger_path:
            return
            
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        rec = bytearray()
        rec.append(event_code & 0xFF)
        rec.extend(int(state_int).to_bytes(6, "big"))
        if intron is not None:
            rec.append(intron & 0xFF)
            
        with open(path, "ab") as f:
            f.write(rec)

    def replay_ledger(self, ledger_path: str) -> None:
        """Reconstruct channels and phase from ledger trajectory."""
        if not os.path.exists(ledger_path):
            return
        
        from baby.kernel.governance import apply_gyration_and_transform
        
        with open(ledger_path, "rb") as f:
            data = f.read()
        
        pos = 0
        current_state = self.start_state()
        
        while pos < len(data):
            event_code = data[pos]
            pos += 1
            state_bytes = data[pos:pos+6]
            state_int = int.from_bytes(state_bytes, "big")
            pos += 6
            intron = data[pos] if pos < len(data) else None
            pos += 1 if intron is not None else 0
            
            if event_code == 0x10:  # Init
                current_state = state_int
            elif event_code in (0x01, 0x02):  # Egress/Emission
                if intron is not None:
                    current_state = apply_gyration_and_transform(current_state, intron)
                    # Rebuild channels/orbit_phase by simulating learn/emission
                    # This is a simplified reconstruction - in practice you'd want more sophisticated rebuilding
                    pass

    # Legacy compatibility
    @property
    def keys(self):
        return self.ontology_keys
    
    @property
    def ep(self):
        return self.epistemology
    
    @property
    def pheno(self):
        return self.phenomenology_map
    
    @property
    def rep_phase(self):
        return self.orbit_phase
    
    @property
    def rep_channel(self):
        return self.slab_channels
    
    def orbit_rep_index(self, index: int) -> int:
        return self.get_orbit_representative(index)
    
    def apply_intron_index(self, index: int, intron: int) -> int:
        return self.apply_intron_to_index(index, intron)
    
    def _state_phase(self, state_int: int) -> Tuple[int, int]:
        return self.compute_state_phase(state_int)
    
    def token_phase(self, token_id: int) -> Tuple[int, int]:
        return self.compute_token_phase(token_id)
    
    def _fold8(self, a: int, b: int) -> Tuple[int, int]:
        return self._fold_with_amplitude(a, b)
    
    def _alignment_amp(self, state_int: int, rep_idx: int) -> int:
        return self.compute_alignment_amplitude(state_int, rep_idx)
    
    def _slab_byte(self, state_int: int, slab_idx: int) -> int:
        return self.extract_slab_byte(state_int, slab_idx)
    
    def sector(self, state_int: int) -> int:
        return self.compute_sector_signature(state_int)
    
    def emit_next(self, index: int, *args, **kwargs):
        return self.emit_next_token(index, *args, **kwargs)