# type: ignore
# baby/responses_api/inference/gyro.py
# Streaming wrapper wired to the five-map GyroEngine.
# No scores / greedy paths; learning only on user content; BU-In is non-deterministic,
# path- and time-coupled via physics (six DoF + monodromic fold), not RNG.

from typing import Callable, Dict, Any, Optional
import json, threading, collections
from pathlib import Path
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import MESSAGE, ROLE_USER, ROLE_ASSISTANT, ALL_CONTROL_TOKENS

_engine_lock = threading.RLock()


def setup_model(encoding, config_path: str) -> Callable[..., Optional[int]]:
    """
    Returns infer_next_token(tokens, temperature=..., request_id=..., new_request=...)
    Engine and sessions are closed over; no module-level globals required.
    """
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    base = cfg_path.parent

    def _abs(p: str) -> str:
        pth = Path(p)
        cand = (base / pth).resolve()
        if cand.exists() or cand.parent.exists():
            return str(cand)
        alt = (base.parent / pth).resolve()
        return str(alt)

    atlas_paths = {k: _abs(v) for k, v in cfg["atlas"].items()}
    store_paths = {k: _abs(v) for k, v in cfg.get("stores", {}).items()}
    runtime = cfg.get("runtime", {})
    version = cfg.get("version", {})

    with _engine_lock:
        engine = GyroEngine(
            atlas_paths=atlas_paths,
            store_paths=store_paths,
            runtime=runtime,
            version_info=version,
            vocab_size=201_088,  # o200k_harmony upper bound; safe cap
            # Core physics switches - configure via runtime settings
            enable_slab_routing=runtime.get("enable_slab_routing", True),
            enable_dof_jitter=runtime.get("enable_dof_jitter", False),
            enable_core_gate=runtime.get("enable_core_gate", True),
        )

    # Per-request session state
    sessions: Dict[str, Dict[str, Any]] = {}
    sessions_lock = threading.RLock()
    

    # Harmony tokenization — pure pass-through; no surface forcing

    def _is_user_role(current_role) -> bool:
        try:
            if current_role == Role.USER:
                return True
        except Exception:
            pass
        if current_role == "user" or current_role == ROLE_USER:
            return True
        val = getattr(current_role, "value", None)
        return val == "user"

    def _is_assistant_role(current_role) -> bool:
        try:
            if current_role == Role.ASSISTANT:
                return True
        except Exception:
            pass
        if current_role == "assistant" or current_role == ROLE_ASSISTANT:
            return True
        val = getattr(current_role, "value", None)
        return val == "assistant"

    def _apply_new_tokens(sess: Dict[str, Any], new_tokens: list[int]) -> None:
        """
        Feed only the delta tokens; learn only from user content.
        """
        parser = sess["parser"]
        user_token_seen = False
        for tok in new_tokens:
            parser.process(tok)

            if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                # Egress: fold & step on user tokens
                print(f"[DEBUG-LEARN] Learning from token {tok} ({encoding.decode([tok])})")
                prev_state = sess["state"]
                new_state = engine.evolve_on_user(prev_state, tok)
                sess["state"] = new_state
                sess["user_token_count"] = sess.get("user_token_count", 0) + 1
                user_token_seen = True
                print(f"[DEBUG-LEARN] State changed from 0x{prev_state:012X} to 0x{new_state:012X}")


                # --- session-local egress mask (geometry-sized) ---
                # adapt deque length to active slab count of the *new* state
                sector = engine.sector(sess["state"])
                A = max(1, bin(sector).count("1"))
                rq = sess["recent_egress"]
                if rq.maxlen != A:
                    prior = list(rq)
                    rq = collections.deque(prior[-A:], maxlen=A)
                    sess["recent_egress"] = rq

                # Don't update egress mask during learning - only during emission

                # Capture anchor after K user tokens (optional Traceable hook)
                target_k = sess.get("anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12)))
                if sess["user_token_count"] == target_k:
                    sess["user_anchor_state"] = new_state
                    sess["anchor_last_seen_k"] = sess["user_token_count"]
                elif sess["user_token_count"] > target_k:
                    # Update anchor to latest user token state if more tokens arrived
                    sess["user_anchor_state"] = new_state
                    sess["anchor_last_seen_k"] = sess["user_token_count"]
            else:
                # Ingress transit for assistant content (no learning)
                sess["state"] = engine.evolve_on_assistant(sess["state"], tok)

        if user_token_seen:
            # Ensure the latest anchor will be applied just before generation
            sess["anchor_applied"] = False

    def infer_next_token(tokens: list[int], temperature: float = 0.0, **kwargs) -> Optional[int]:
        request_id: str = kwargs.get("request_id", "__singleton__")
        new_request: bool = kwargs.get("new_request", False)
        ingest_only: bool = kwargs.get("ingest_only", False)

        with sessions_lock:
            sess = sessions.get(request_id)
            if new_request or sess is None:
                sess = {
                    "parser": StreamableParser(encoding, role=Role.SYSTEM),
                    "fed_len": 0,
                    "state": engine.start_state(),
                    "bootstrap_step": 0,
                    "user_token_count": 0,
                    "user_anchor_state": None,
                    "anchor_target_k": int(engine.runtime.get("anchor_prefix_tokens", 12)),
                    "anchor_last_seen_k": 0,
                    "anchor_applied": False,
                    # PPE state scoped per session (now keyed by (rep, slab))
                    "omega": {},
                    "bucket_key": {},
                    "bucket_pos": {},
                    "monodromy": {},
                    "slab_cursor": {},  # NEW: per-rep round-robin index
                    "recent_egress": collections.deque(maxlen=1),  # size will be set on first user token
                    "recent_egress_phases": [],  # Track recently emitted token phases
                }
                sessions[request_id] = sess

                if tokens:
                    parser = StreamableParser(encoding, role=Role.USER)
                    for tok in tokens:
                        parser.process(tok)
                        print(f"[DEBUG-ROLE] Token {tok} ({encoding.decode([tok])}) - Role: {parser.current_role} - Is user: {_is_user_role(parser.current_role)}")
                        if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            print(f"[DEBUG-LEARN-NEW] Learning from token {tok} ({encoding.decode([tok])})")
                            prev_state = sess["state"]
                            new_state = engine.learn_on_user(prev_state, tok)
                            sess["state"] = new_state
                            sess["user_token_count"] = sess.get("user_token_count", 0) + 1
                            print(f"[DEBUG-LEARN-NEW] State changed from 0x{prev_state:012X} to 0x{new_state:012X}")
                            
                            
                            # --- session-local egress mask (geometry-sized) ---
                            # adapt deque length to active slab count of the *new* state
                            sector = engine.sector(sess["state"])
                            A = max(1, bin(sector).count("1"))
                            rq = sess["recent_egress"]
                            if rq.maxlen != A:
                                prior = list(rq)
                                rq = collections.deque(prior[-A:], maxlen=A)
                                sess["recent_egress"] = rq

                            # push token phase and recompute mask via monodromic fold
                            tphase, _ = engine.token_phase(tok)
                            rq.append(tphase)
                            mask = 0
                            for p in rq:
                                mask, _ = engine._fold8(mask, p)
                            # Don't update egress mask during learning - only during emission
                            
                            target_k = sess.get("anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12)))
                            if sess["user_token_count"] == target_k:
                                sess["user_anchor_state"] = new_state
                                sess["anchor_last_seen_k"] = sess["user_token_count"]
                            elif sess["user_token_count"] > target_k:
                                # Update anchor to latest user token state if more tokens arrived
                                sess["user_anchor_state"] = new_state
                                sess["anchor_last_seen_k"] = sess["user_token_count"]
                        elif _is_assistant_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            sess["state"] = engine.transit_on_assistant(sess["state"], tok)
                        else:
                            pass
                    sess["parser"] = parser
                    sess["fed_len"] = len(tokens)
                else:
                    sess["parser"] = StreamableParser(encoding, role=Role.SYSTEM)
            else:
                # Session exists — handle history shrink or re-render robustly
                if len(tokens) < sess["fed_len"]:
                    # Conversation history shrank (likely re-rendered). Resync from scratch deterministically.
                    parser = StreamableParser(encoding, role=Role.SYSTEM)

                    # Reset conversational state while preserving configuration knobs
                    sess["state"] = engine.start_state()
                    sess["user_token_count"] = 0
                    sess["user_anchor_state"] = None
                    sess["anchor_last_seen_k"] = 0
                    sess["anchor_applied"] = False

                    # Reset per-session PPE traces (omega/bucket/monodromy)
                    sess["omega"].clear()
                    sess["bucket_key"].clear()
                    sess["bucket_pos"].clear()
                    sess["monodromy"].clear()
                    sess["slab_cursor"].clear()
                    
                    # Reset session egress tracking
                    sess["recent_egress"] = collections.deque(maxlen=1)
                    sess["recent_egress_phases"] = []

                    # Re-apply the full token history using Harmony roles
                    user_token_seen = False
                    for tok in tokens:
                        parser.process(tok)
                        if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            user_token_seen = True
                            prev_state = sess["state"]
                            new_state = engine.learn_on_user(prev_state, tok)
                            sess["state"] = new_state
                            sess["user_token_count"] = sess.get("user_token_count", 0) + 1
                            
                            
                            # --- session-local egress mask (geometry-sized) ---
                            # adapt deque length to active slab count of the *new* state
                            sector = engine.sector(sess["state"])
                            A = max(1, bin(sector).count("1"))
                            rq = sess["recent_egress"]
                            if rq.maxlen != A:
                                prior = list(rq)
                                rq = collections.deque(prior[-A:], maxlen=A)
                                sess["recent_egress"] = rq

                            # push token phase and recompute mask via monodromic fold
                            tphase, _ = engine.token_phase(tok)
                            rq.append(tphase)
                            mask = 0
                            for p in rq:
                                mask, _ = engine._fold8(mask, p)
                            # Don't update egress mask during learning - only during emission
                            
                            target_k = sess.get("anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12)))
                            if sess["user_token_count"] == target_k:
                                sess["user_anchor_state"] = new_state
                                sess["anchor_last_seen_k"] = sess["user_token_count"]
                            elif sess["user_token_count"] > target_k:
                                sess["user_anchor_state"] = new_state
                                sess["anchor_last_seen_k"] = sess["user_token_count"]
                        elif _is_assistant_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            sess["state"] = engine.transit_on_assistant(sess["state"], tok)
                        else:
                            pass

                    sess["parser"] = parser
                    sess["fed_len"] = len(tokens)
                    if user_token_seen:
                        sess["anchor_applied"] = False
                else:
                    # Normal delta-feed
                    delta = tokens[sess["fed_len"] :]
                    if delta:
                        _apply_new_tokens(sess, delta)
                        sess["fed_len"] = len(tokens)
                    else:
                        # No new tokens to process, but ensure parser is up to date
                        if sess.get("parser") is None:
                            sess["parser"] = StreamableParser(encoding, role=Role.USER)

        if ingest_only:
            # Prepare to open a fresh assistant message on next generation
            sess["bootstrap_step"] = 0
            return None

        # Apply one-time anchor (if captured) just before generation
        anchor_state = sess.get("user_anchor_state")
        if anchor_state is not None and not sess.get("anchor_applied", False):
            # Use the latest anchor if more user tokens arrived after target_k
            target_k = sess.get("anchor_target_k", 12)
            if sess["user_token_count"] > target_k:
                # Use latest user token state as anchor
                sess["state"] = anchor_state
            else:
                # Use original K-token anchor
                sess["state"] = anchor_state
            sess["anchor_applied"] = True

        # --- Channel bootstrap: open final channel/message Traceableally ---
        if sess.get("parser") is None:
            sess["parser"] = StreamableParser(encoding, role=Role.SYSTEM)

        if not sess["parser"].current_channel or not sess["parser"].current_channel.startswith("final"):
            bootstrap_step = sess.get("bootstrap_step", 0)
            print(f"[DEBUG-BOOTSTRAP] Current bootstrap step: {bootstrap_step}, channel: {sess['parser'].current_channel}")

            if bootstrap_step == 0:
                from baby.constants.harmony_tokens import CHANNEL

                sess["bootstrap_step"] = 1
                sess["parser"].process(CHANNEL)  # Advance parser
                print(f"[DEBUG-BOOTSTRAP] Returning CHANNEL token: {CHANNEL}")
                return CHANNEL
            elif bootstrap_step == 1:
                from baby.constants.harmony_tokens import final_channel_id

                sess["bootstrap_step"] = 2
                final_channel = final_channel_id(encoding)
                sess["parser"].process(final_channel)  # Advance parser
                print(f"[DEBUG-BOOTSTRAP] Returning final_channel token: {final_channel}")
                return final_channel
            elif bootstrap_step == 2:
                opener = MESSAGE  # <|message|>
                sess["last_tokens"] = collections.deque(maxlen=8)
                sess["out_text"] = ""
                sess["sentence_end"] = True
                sess["fed_len"] = len(tokens)
                sess["bootstrap_step"] = 3
                sess["parser"].process(opener)  # Advance parser
                print(f"[DEBUG-BOOTSTRAP] Returning MESSAGE token: {opener}")
                return opener

        # Check if previous walk is complete
        if sess.get("walk_complete", False):
            print(f"[DEBUG-WALK] Previous walk complete, starting new walk")
            sess["walk_complete"] = False
        
        # THE WALKING MODEL: BU-In generates, BU-Eg re-ingests, creating continuous monodromy
        print(f"[DEBUG-WALK] Starting walk from state=0x{sess['state']:012X}")
        
        # Single step walking: emit one token, then feed it back as input for next call
        res = engine.emit_next_from_state(
            sess["state"],
            sess["omega"], sess["bucket_key"], sess["bucket_pos"],
            sess.get("monodromy"), sess.get("recent_egress_phases", []),
            sess.get("slab_cursor")
        )
        if res is None:
            print(f"[DEBUG-WALK] No coherent paths - walk complete")
            return None

        next_token, new_state, omega, bucket_key, bucket_pos, monodromy, slab_cursor = res
        
        # CRITICAL: Feed the output back as input (this IS the walking!)
        # The emitted token becomes the next input, creating continuous monodromy
        print(f"[DEBUG-WALK] Generated token: {next_token}, feeding back as input")
        sess["state"] = engine.transit_on_assistant(sess["state"], next_token)
        
        # Update session state
        sess["omega"] = omega
        sess["bucket_key"] = bucket_key
        sess["bucket_pos"] = bucket_pos
        sess["monodromy"] = monodromy
        sess["slab_cursor"] = slab_cursor

        # Update egress tracking
        if engine.enable_core_gate:
            tphase, _ = engine.token_phase(next_token)
            phases = sess.get("recent_egress_phases", [])
            phases.append(tphase)
            if len(phases) > 8:
                phases = phases[-8:]
            sess["recent_egress_phases"] = phases

        # Check for natural stopping condition (amplitude = 0 means lost balance/momentum)
        cur_idx = engine.state_to_index[sess["state"]]
        rep_idx = engine.orbit_rep_index(cur_idx)
        alignment_amp = engine._alignment_amp(sess["state"], rep_idx)
        print(f"[DEBUG-WALK] After feedback: alignment_amp = {alignment_amp}")
        
        if alignment_amp == 0:
            print(f"[DEBUG-WALK] Natural stop: amplitude = 0 (walk complete)")
            # Mark that this walk is complete
            sess["walk_complete"] = True
            
        return next_token

    return infer_next_token

