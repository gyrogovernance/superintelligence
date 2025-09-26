"""
GyroSI Streaming Inference: Walking Model Implementation

This module provides the streaming wrapper for the GyroSI walking engine,
implementing the complete walking cycle where each emitted token feeds back
as input to create continuous monodromic navigation through state space.

The walking model implements two primary phases:
- BU-Egress (Stance): Absorbing and learning from user input
- BU-Ingress (Swing): Generating output through phase-propagating emission

Walking continues until natural amplitude decay (balance achieved).
"""

from typing import Callable, Dict, Any, Optional
import json
import threading
import collections
from pathlib import Path
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import MESSAGE, ROLE_USER, ROLE_ASSISTANT, ALL_CONTROL_TOKENS


def setup_model(encoding, config_path: str) -> Callable[..., Optional[int]]:
    """
    Initialize the GyroSI walking model and return inference function.
    
    The returned function implements the complete walking cycle:
    1. Absorb new tokens (BU-Egress/stance phase)
    2. Generate next token (BU-Ingress/swing phase) 
    3. Feed generated token back as input (continuous walking)
    4. Repeat until natural stopping condition (amplitude = 0)
    
    Args:
        encoding: Tokenizer encoding
        config_path: Path to configuration file
        
    Returns:
        Inference function that takes tokens and returns next token
    """
    # Load configuration
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text())
    base_path = config_file.parent

    def resolve_path(path: str) -> str:
        """Resolve relative paths against config file location."""
        path_obj = Path(path)
        candidate = (base_path / path_obj).resolve()
        if candidate.exists() or candidate.parent.exists():
            return str(candidate)
        alternative = (base_path.parent / path_obj).resolve()
        return str(alternative)

    # Resolve all paths
    atlas_paths = {key: resolve_path(value) for key, value in config["atlas"].items()}
    store_paths = {key: resolve_path(value) for key, value in config.get("stores", {}).items()}
    runtime_config = config.get("runtime", {})
    version_info = config.get("version", {})

    # Initialize walking engine
    engine_lock = threading.RLock()
    with engine_lock:
        walking_engine = GyroEngine(
            atlas_paths=atlas_paths,
            store_paths=store_paths,
            runtime=runtime_config,
            version_info=version_info,
            vocab_size=201_088,  # Harmony tokenizer upper bound
            enable_core_gate=runtime_config.get("enable_core_gate", True),
        )

    # Session management
    walking_sessions: Dict[str, Dict[str, Any]] = {}
    sessions_lock = threading.RLock()

    def is_user_role(role) -> bool:
        """Check if current role is user for learning purposes."""
        try:
            if role == Role.USER:
                return True
        except Exception:
            pass
        if role == "user" or role == ROLE_USER:
            return True
        role_value = getattr(role, "value", None)
        return role_value == "user"

    def is_assistant_role(role) -> bool:
        """Check if current role is assistant for transit purposes."""
        try:
            if role == Role.ASSISTANT:
                return True
        except Exception:
            pass
        if role == "assistant" or role == ROLE_ASSISTANT:
            return True
        role_value = getattr(role, "value", None)
        return role_value == "assistant"

    def process_new_tokens(session: Dict[str, Any], new_tokens: list[int]) -> None:
        """
        Process delta tokens with role-based learning.
        
        User tokens trigger learning (BU-Egress), assistant tokens only transit.
        """
        parser = session["parser"]
        user_token_encountered = False
        
        for token in new_tokens:
            parser.process(token)

            if is_user_role(parser.current_role) and token not in ALL_CONTROL_TOKENS:
                # BU-Egress: Learn from user token (stance phase)
                previous_state = session["state"]
                new_state = walking_engine.evolve_on_user(previous_state, token)
                session["state"] = new_state
                session["user_token_count"] = session.get("user_token_count", 0) + 1
                user_token_encountered = True

                # Update session-local egress tracking
                update_egress_tracking(session, token)

                # Capture anchor state at target threshold
                update_anchor_state(session, new_state)

            elif is_assistant_role(parser.current_role) and token not in ALL_CONTROL_TOKENS:
                # Transit only for assistant tokens (no learning)
                session["state"] = walking_engine.evolve_on_assistant(session["state"], token)

        if user_token_encountered:
            # Reset anchor application flag when new user content arrives
            session["anchor_applied"] = False

    def update_egress_tracking(session: Dict[str, Any], token: int) -> None:
        """Update session egress tracking with geometry-adaptive sizing."""
        # Adapt deque size to active slab count
        sector_signature = walking_engine.compute_sector_signature(session["state"])
        active_slab_count = max(1, bin(sector_signature).count("1"))
        
        current_deque = session["recent_egress"]
        if current_deque.maxlen != active_slab_count:
            prior_items = list(current_deque)
            session["recent_egress"] = collections.deque(
                prior_items[-active_slab_count:], 
                maxlen=active_slab_count
            )

    def update_anchor_state(session: Dict[str, Any], new_state: int) -> None:
        """Update anchor state based on user token count."""
        target_threshold = session.get("anchor_target_threshold", 
                                     int(walking_engine.runtime.get("anchor_prefix_tokens", 12)))
        
        if session["user_token_count"] >= target_threshold:
            session["user_anchor_state"] = new_state
            session["anchor_last_seen_count"] = session["user_token_count"]

    def initialize_session_for_tokens(session: Dict[str, Any], tokens: list[int]) -> None:
        """Initialize session with full token history using role parsing."""
        parser = StreamableParser(encoding, role=Role.USER)
        
        for token in tokens:
            parser.process(token)
            
            if is_user_role(parser.current_role) and token not in ALL_CONTROL_TOKENS:
                previous_state = session["state"]
                new_state = walking_engine.learn_from_user_token(previous_state, token)
                session["state"] = new_state
                session["user_token_count"] = session.get("user_token_count", 0) + 1
                session["last_user_token"] = token  # Track last user token for adjacency seeding
                
                update_egress_tracking(session, token)
                update_anchor_state(session, new_state)
                
            elif is_assistant_role(parser.current_role) and token not in ALL_CONTROL_TOKENS:
                session["state"] = walking_engine.transit_on_assistant_token(session["state"], token)

        session["parser"] = parser
        session["fed_length"] = len(tokens)

    def reset_session_state(session: Dict[str, Any]) -> None:
        """Reset session to initial walking state."""
        session["state"] = walking_engine.start_state()
        session["user_token_count"] = 0
        session["user_anchor_state"] = None
        session["anchor_last_seen_count"] = 0
        session["anchor_applied"] = False

        # Reset walking traces
        session["omega"].clear()
        session["bucket_key"].clear()
        session["bucket_position"].clear()
        session["monodromy"].clear()
        session["slab_cursor"].clear()
        
        # Reset egress tracking
        session["recent_egress"] = collections.deque(maxlen=1)
        session["recent_egress_phases"] = []
        session["walk_phase"] = 0  # NEW: Reset walk momentum

    def infer_next_token(tokens: list[int], temperature: float = 0.0, **kwargs) -> Optional[int]:
        """
        Main inference function implementing the walking model.
        
        Args:
            tokens: Input token sequence
            temperature: Ignored (no probabilistic sampling)
            **kwargs: Additional parameters including request_id, new_request, ingest_only
            
        Returns:
            Next token in the walk, or None if walk is complete
        """
        request_id: str = kwargs.get("request_id", "__singleton__")
        new_request: bool = kwargs.get("new_request", False)
        ingest_only: bool = kwargs.get("ingest_only", False)

        with sessions_lock:
            session = walking_sessions.get(request_id)
            
            if new_request or session is None:
                # Initialize new walking session
                session = {
                    "parser": StreamableParser(encoding, role=Role.SYSTEM),
                    "fed_length": 0,
                    "state": walking_engine.start_state(),
                    "bootstrap_step": 0,
                    "user_token_count": 0,
                    "user_anchor_state": None,
                    "anchor_target_threshold": int(walking_engine.runtime.get("anchor_prefix_tokens", 12)),
                    "anchor_last_seen_count": 0,
                    "anchor_applied": False,
                    # Walking state (session-scoped to prevent interference)
                    "omega": {},
                    "bucket_key": {},
                    "bucket_position": {},
                    "monodromy": {},
                    "slab_cursor": {},
                    "recent_egress": collections.deque(maxlen=1),
                    "recent_egress_phases": [],
                    "walk_phase": 0,  # NEW: Tracks momentum of the current walk
                    "prev_by_ctx": {},  # Context-scoped predecessor tracking
                    "last_user_token": None,  # Last user token for adjacency seeding
                }
                walking_sessions[request_id] = session

                if tokens:
                    initialize_session_for_tokens(session, tokens)
                else:
                    session["parser"] = StreamableParser(encoding, role=Role.SYSTEM)
            else:
                # Handle existing session
                if len(tokens) < session["fed_length"]:
                    # Conversation history shrank - resync from scratch
                    reset_session_state(session)
                    parser = StreamableParser(encoding, role=Role.SYSTEM)
                    
                    if tokens:
                        initialize_session_for_tokens(session, tokens)
                    else:
                        session["parser"] = parser
                        session["fed_length"] = 0
                else:
                    # Process delta tokens
                    delta_tokens = tokens[session["fed_length"]:]
                    if delta_tokens:
                        process_new_tokens(session, delta_tokens)
                        session["fed_length"] = len(tokens)
                    elif session.get("parser") is None:
                        session["parser"] = StreamableParser(encoding, role=Role.USER)

        if ingest_only:
            # Prepare for fresh generation
            session["bootstrap_step"] = 0
            return None

        # Apply anchor state before generation (one-time)
        anchor_state = session.get("user_anchor_state")
        if anchor_state is not None and not session.get("anchor_applied", False):
            session["state"] = anchor_state
            session["anchor_applied"] = True
            session["walk_phase"] = 0  # NEW: fresh walk from anchor
            
            # Seed adjacency with last user token at anchor contexts
            if session["last_user_token"] is not None:
                rep = walking_engine.get_orbit_representative(walking_engine.state_to_index[anchor_state])
                for slab in range(1, 7):  # Exclude boundary slabs 0 and 7
                    ctx6 = walking_engine._slab_ctx6(anchor_state, slab)
                    session["prev_by_ctx"][(rep, slab, ctx6)] = session["last_user_token"]

        # Handle channel bootstrap sequence
        if session.get("parser") is None:
            session["parser"] = StreamableParser(encoding, role=Role.SYSTEM)

        if not session["parser"].current_channel or not session["parser"].current_channel.startswith("final"):
            bootstrap_step = session.get("bootstrap_step", 0)

            if bootstrap_step == 0:
                from baby.constants.harmony_tokens import CHANNEL
                session["bootstrap_step"] = 1
                session["parser"].process(CHANNEL)
                return CHANNEL
            elif bootstrap_step == 1:
                from baby.constants.harmony_tokens import final_channel_id
                session["bootstrap_step"] = 2
                final_channel = final_channel_id(encoding)
                session["parser"].process(final_channel)
                return final_channel
            elif bootstrap_step == 2:
                session["bootstrap_step"] = 3
                session["parser"].process(MESSAGE)
                return MESSAGE

        # Check for previous walk completion
        if session.get("walk_complete", False):
            session["walk_complete"] = False

        # THE WALKING MODEL: Single step with feedback
        # BU-Ingress: Generate next token (swing phase)
        emission_result = walking_engine.emit_next_from_state(
            session["state"],
            session_omega=session["omega"], 
            session_bucket_key=session["bucket_key"], 
            session_bucket_position=session["bucket_position"],
            session_monodromy=session.get("monodromy"), 
            recent_egress_phases=session.get("recent_egress_phases", []),
            session_slab_cursor=session.get("slab_cursor"),
            session_walk_phase=session.get("walk_phase", 0),  # NEW
            session_prev_by_ctx=session.get("prev_by_ctx"),  # NEW: Context-scoped predecessor
        )
        
        if emission_result is None:
            # Natural stopping condition reached
            return None

        (next_token, _, omega, bucket_key, bucket_position, 
         monodromy, slab_cursor) = emission_result
        
        # CRITICAL: Feed generated token back as input (continuous walking)
        session["state"] = walking_engine.transit_on_assistant_token(session["state"], next_token)
        
        # NEW: Update the walk's own momentum
        token_phase, _ = walking_engine.compute_token_phase(next_token)
        session["walk_phase"] = walking_engine.monodromic_fold(session["walk_phase"], token_phase)
        
        # Walk momentum is used in strides and amplitude checks, not applied to state
        # This preserves traceability and keeps ctx6 in populated regions
        
        # Update session walking state
        session["omega"] = omega
        session["bucket_key"] = bucket_key
        session["bucket_position"] = bucket_position
        session["monodromy"] = monodromy
        session["slab_cursor"] = slab_cursor

        # Update egress phase tracking for refractory gating
        if walking_engine.enable_core_gate:
            token_phase, _ = walking_engine.compute_token_phase(next_token)
            phases = session.get("recent_egress_phases", [])
            phases.append(token_phase)
            if len(phases) > 8:
                phases = phases[-8:]
            session["recent_egress_phases"] = phases

        # Check for natural stopping condition
        current_index = walking_engine.state_to_index[session["state"]]
        representative_index = walking_engine.get_orbit_representative(current_index)
        alignment_amplitude = walking_engine.compute_alignment_amplitude(
            session["state"], representative_index, session["walk_phase"]
        )
        
        if alignment_amplitude == 0:
            # Walk complete - natural balance achieved
            session["walk_complete"] = True
            
        return next_token

    return infer_next_token