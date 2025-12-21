# type: ignore
# baby/kernel/gyro_core.py
# Minimal five-map runtime core for GyroSI.
# Uses ONLY the canonical atlas artifacts:
#   - theta.npy                (CS)
#   - ontology_keys.npy        (UNA)
#   - epistemology.npy         (BU-Eg transitions)
#   - phenomenology_map.npy    (ONA canonical orbits)
#   - orbit_sizes.npy          (BU-In cardinalities)
#
# Pure monodromic unfold BU-In:
#   BU-Eg (user): fold token intron path -> token_phase; update per-orbit phase:
#                 new_phase = fold(rep_phase, token_phase); register token in
#                 rep_channel[rep][new_phase].
#   BU-In (emit): compute state_phase; pick one of the learned phases
        #                 coherently from the set of keys; emit from its bucket.
#   No scoring, no admissibility filters, no recovery ladders.

import numpy as np
import pickle
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import gcd

# ---------- ψ boundary ----------


def byte_to_intron(b: int) -> int:  # ψ
    return (b & 0xFF) ^ 0xAA


def intron_to_byte(i: int) -> int:  # ψ⁻¹
    return (i & 0xFF) ^ 0xAA


def token_to_introns(token_id: int) -> List[int]:
    """
    Harmony ids are integers; represent in big-endian bytes, then apply ψ.
    """
    if token_id < 0:
        raise ValueError("Negative token id")
    bs = token_id.to_bytes((token_id.bit_length() + 7) // 8 or 1, "big")
    return [byte_to_intron(b) for b in bs]


# ---------- GyroEngine with 5 maps ↔ 5 stages ----------


class GyroEngine:
    """
    Stages → Maps
      1) CS      → theta.npy              (divergence from archetype)
      2) UNA     → ontology_keys.npy      (the discovered manifold)
      3) ONA     → phenomenology_map.npy  (canonical orbit representative)
      4) BU-Eg   → epistemology.npy       (state transition by intron)
       5) BU-In   → orbit_sizes.npy        (cardinality for traceable ordering)
    """

    def __init__(
        self,
        atlas_paths: Dict[str, str],
        store_paths: Optional[Dict[str, str]] = None,
        runtime: Optional[Dict[str, str]] = None,
        version_info: Optional[Dict[str, str]] = None,
        vocab_size: int = 201_088,
        # Core physics switches - disable all secondary heuristics for testing
        enable_slab_routing: bool = False,
        enable_dof_jitter: bool = False,
        enable_core_gate: bool = False,
    ):
        # Required five maps
        self.theta = np.load(atlas_paths["theta"], mmap_mode="r")  # float32[N]
        self.keys = np.load(atlas_paths["ontology_keys"], mmap_mode="r")  # uint64[N]
        self.ep = np.load(atlas_paths["epistemology"], mmap_mode="r")  # int32 [N,256]
        self.pheno = np.load(atlas_paths["phenomenology_map"], mmap_mode="r")  # int32 [N]
        self.orbit_sizes = np.load(atlas_paths["orbit_sizes"], mmap_mode="r")  # uint32[N]

        # Reverse index: state_int → index (canonical)
        self.state_to_index: Dict[int, int] = {int(s): int(i) for i, s in enumerate(self.keys)}

        # Canonical orbit representatives (indices)
        reps = np.unique(self.pheno)
        self.orbit_reps: List[int] = [int(r) for r in reps]  # typically 256

        # --- Pure monodromic BU-In state ---
        # Per-orbit phase accumulator (updated only when user speaks)
        self.rep_phase: Dict[int, int] = {}  # rep_idx -> 8-bit phase (0..255)
        # Token channels per orbit keyed by CUMULATIVE phase reached after learning
        # Now supports slab-specific channels: (rep_idx, slab_idx) -> { phase -> [token_id, ...] }
        self.rep_channel: Dict[Tuple[int, int], Dict[int, List[int]]] = (
            {}
        )  # (rep_idx, slab_idx) -> { phase_after_learning (0..255) -> [token_id, ...] }

        # --- Emission hygiene (path-native, non-competitive) ---
        self.emit_gate: Dict[int, int] = {}        # per-rep monodromic refractory gate (8-bit)
        self.last_token: Dict[int, int] = {}       # last emitted token per rep
        self.last_emit_tick: Dict[int, int] = {}   # last free_tick per rep

        # --- Phase-Propagating Emission (PPE) state ---
        # PPE state is now managed at session level to prevent concurrent session bleeding

        # Passive diagnostics removed (unused)

        # Concurrency protection
        self._lock = threading.RLock()

        # Persistence cadence control
        self._token_counter = 0
        self._last_save_time = time.time()
        self._save_interval_tokens = 100  # Save after N tokens
        self._save_interval_seconds = 30.0  # Save after T seconds
        self._pending_changes = False

        # Bucket capacity discipline
        self._max_bucket_size = 64  # Maximum tokens per bucket (K)

        self.store_paths = store_paths or {}
        self.runtime = runtime or {}
        self.version_info = version_info or {}

        self.vocab_size = vocab_size
        self.vocab_max = vocab_size

        # Core physics switches
        self.enable_slab_routing = enable_slab_routing
        self.enable_dof_jitter = enable_dof_jitter
        self.enable_core_gate = enable_core_gate

        # Extended switches (read from runtime to avoid changing constructor signature)
        # Phase alignment is now intrinsic - no longer optional
        
        # Override core gate from runtime if specified
        if "enable_core_gate" in (self.runtime or {}):
            self.enable_core_gate = bool(self.runtime["enable_core_gate"])


        # Start = argmin θ (phenomenal archetype)
        self.start_index: int = int(np.argmin(self.theta))

        # Ensure store files exist if store paths are provided (create only if missing)
        self._ensure_store_files()

        # Load any existing learned data from disk
        self._load_learned_data()

    # ---------- map access (steps 1–4) ----------

    def theta_of_index(self, idx: int) -> float:
        return float(self.theta[idx])

    def orbit_rep_index(self, idx: int) -> int:
        return int(self.pheno[idx])

    def apply_intron_index(self, idx: int, intron: int) -> int:
        if not (0 <= intron <= 255):
            raise ValueError("intron out of range")
        
        new_idx = int(self.ep[idx, intron])

        # --- Chirality Guard is now a physical law, not an option ---
        state_before = int(self.keys[idx])
        state_after = int(self.keys[new_idx])
        
        # Parity of state change (number of bit flips)
        parity = bin(state_before ^ state_after).count("1") & 1
        
        # An even parity (0) represents a symmetric, non-chiral transition.
        # The physical law of CGM rejects this and applies a corrective gyration.
        if parity == 0:
            # The recovery operation is applying the dual intron (XOR 0xAA),
            # which is the simplest chiral correction.
            recovery_intron = intron ^ 0xAA
            return int(self.ep[idx, recovery_intron])
            
        return new_idx

    def _coprime_stride(self, seed: int, n: int) -> int:
        """Return a stride s in [1..n-1] co-prime with n, derived from seed via fold-compatible arithmetic."""
        if n <= 1:
            return 1
        s = (1 + (seed % n)) or 1
        # enforce co-primeness without thresholds
        # n is small (# of keys in the slab context), loop ≤ n
        while gcd(s, n) != 1:
            s = (s + 1) % n
            if s == 0:
                s = 1
        return s

    def _alignment_amp(self, state_int: int, rep_idx: int) -> int:
        """Amplitude of alignment between live state phase and orbit phase memory."""
        sp, _ = self._state_phase(state_int)
        rp = self.rep_phase.get(rep_idx, 0)
        _, amp = self._fold8(sp, rp)
        return amp

    # ---------- Monodromic Fold (8-bit) ----------

    @staticmethod
    def fold(acc: int, intron: int) -> int:
        """Legacy fold operation for backward compatibility."""
        # a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b)) over 8-bit
        a = acc & 0xFF
        b = intron & 0xFF
        return (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF

    def fold_sequence(self, introns: List[int], acc: int = 0) -> Tuple[int, int]:
        """Fold sequence returning (phase, amplitude) for interference analysis."""
        m = acc & 0xFF
        amp = 0
        for i in introns:
            m, amp = self._fold8(m, i)
        return m, amp

    # ---------- Pure monodromic unfold helpers ----------

    @staticmethod
    def _fold8(a: int, b: int) -> Tuple[int, int]:
        """Fold operation returning (phase, amplitude) for interference analysis."""
        a &= 0xFF
        b &= 0xFF
        res = (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF
        amp = bin(res).count('1')  # Non-zero bits as "coherence strength"
        return res, amp

    def _state_phase(self, state_int: int) -> Tuple[int, int]:
        """
        Project the 48-bit state into an 8-bit phase and velocity by folding its 6 bytes
        after ψ (XOR 0xAA). Returns (phase, velocity) for interference analysis.
        """
        bs = int(state_int).to_bytes(6, "big")
        acc = 0
        prev_acc = 0
        velocity = 0
        for by in bs:
            prev_acc = acc
            acc, _ = self._fold8(acc, by ^ 0xAA)
            velocity, _ = self._fold8(velocity, acc ^ prev_acc)  # Delta as "speed"
        return acc, velocity

    def _bitcount8(self, x: int) -> int:
        """Count set bits in 8-bit integer."""
        return bin(x & 0xFF).count("1")

    def gyro_dist(self, a: int, b: int) -> int:
        """Canonical coherence distance via fold then popcount."""
        comp, _ = self._fold8(a, b)
        return self._bitcount8(comp)

    def _state_phase_components(self, state_int: int) -> Tuple[int, int, int]:
        """
        Compute LI/FG/BG components of the live phase using EXON masks.
        Returns (sp_li, sp_fg, sp_bg) as 8-bit values.
        """
        from baby.kernel.governance import EXON_LI_MASK, EXON_FG_MASK, EXON_BG_MASK

        # Extract 6 bytes from state
        bs = int(state_int).to_bytes(6, "big")

        # Apply masks to each byte and fold separately
        acc_li = 0
        acc_fg = 0
        acc_bg = 0

        for by in bs:
            by_psi = by ^ 0xAA  # Apply ψ transformation
            acc_li, _ = self._fold8(acc_li, by_psi & EXON_LI_MASK)
            acc_fg, _ = self._fold8(acc_fg, by_psi & EXON_FG_MASK)
            acc_bg, _ = self._fold8(acc_bg, by_psi & EXON_BG_MASK)

        return acc_li, acc_fg, acc_bg

    def token_phase(self, token_id: int) -> Tuple[int, int]:
        """Compute token phase and amplitude for interference analysis."""
        return self.fold_sequence(token_to_introns(token_id), 0)

    # ---------- Freedom kernel: six DoF + free-tick ----------

    def _free_tick(self) -> int:
        """
        Endogenous temporal phase: fold 6 bytes of the current monotonic clock
        through ψ into an 8-bit tick. This is *not* RNG; it's the physical time
        boundary coupled to the monodromic fold, giving non-deterministic but
        lawful motion (BU Egress→Ingress).
        """
        ns = time.time_ns()
        bs = ns.to_bytes(8, "big")[-6:]  # 6 bytes → 48-bit affinity
        acc = 0
        for b in bs:
            acc, _ = self._fold8(acc, b ^ 0xAA)  # ψ at the boundary
        return acc  # 0..255

    def _timed_phase(self, phase: int) -> int:
        """
        Apply time twist to phase for endogenous variation.
        Theory: Time is helical parameter (SU(2) progression) - always used for lawful variation.
        """
        tick = self._free_tick()
        timed, _ = self._fold8(phase, tick)
        return timed

    def _row_parity_fold(self, state_int: int, row: int, frame: Optional[int] = None) -> int:
        """
        Fold parity of a given tensor row across all (layer, [frame], col).
        If frame is None, both frames contribute; otherwise only that frame.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        acc = 0
        for layer in range(FROZEN_CHANNELS.NUM_LAYERS):  # 4
            for fr in (range(FROZEN_CHANNELS.NUM_FRAMES) if frame is None else [frame]):  # 2 or 1
                for col in range(FROZEN_CHANNELS.NUM_COLS):  # 2
                    bit_idx = FROZEN_CHANNELS.get_bit_index(layer, fr, row, col)
                    bit = (state_int >> bit_idx) & 1
                    acc, _ = self._fold8(acc, (bit & 1) * 0x01)  # compress parity into 8-bit phase
        return acc

    def _six_dof(self, state_int: int) -> Tuple[int, int, int, int, int, int]:
        """
        Six freedoms (3 rotational + 3 translational) as 8-bit phases:

        - Rotational (rX, rY, rZ): per-row parity over *both* frames (frame-summed)
        - Translational (tX, tY, tZ): per-row *frame-difference* parity (frame 0 vs 1)

        Rows 0/1/2 correspond to the three spatial axes from GENE_Com_S.
        """
        # rotational: both frames together
        rX = self._row_parity_fold(state_int, row=0, frame=None)
        rY = self._row_parity_fold(state_int, row=1, frame=None)
        rZ = self._row_parity_fold(state_int, row=2, frame=None)

        # translational: difference between frames
        f0X = self._row_parity_fold(state_int, row=0, frame=0)
        f1X = self._row_parity_fold(state_int, row=0, frame=1)
        f0Y = self._row_parity_fold(state_int, row=1, frame=0)
        f1Y = self._row_parity_fold(state_int, row=1, frame=1)
        f0Z = self._row_parity_fold(state_int, row=2, frame=0)
        f1Z = self._row_parity_fold(state_int, row=2, frame=1)

        tX, _ = self._fold8(f0X, f1X)  # path-dependent difference via fold
        tY, _ = self._fold8(f0Y, f1Y)
        tZ, _ = self._fold8(f0Z, f1Z)

        return rX, rY, rZ, tX, tY, tZ

    def _compute_velocity_offset(self, state_int: int) -> int:
        """
        Compute velocity from 6 DoF as a compact momentum-like 8-bit value.
        """
        rX, rY, rZ, tX, tY, tZ = self._six_dof(state_int)
        velocity = 0
        for d in (rX, rY, rZ, tX, tY, tZ):
            velocity, _ = self._fold8(velocity, d)
        return velocity

    def _slab_byte(self, state_int: int, slab_idx: int) -> int:
        """
        Compress the 6 bits of a slab into the low bits of a byte (contiguous),
        then apply ψ. This fixes the previous shift-by-min-index approach,
        which preserved gaps and bled geometry.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
        b = 0
        for j, bit_idx in enumerate(indices):
            b |= ((state_int >> bit_idx) & 1) << j
        return (b ^ 0xAA) & 0xFF  # ψ

    # Address helpers removed (unused)

    # ---------- Egress (BU-Eg): absorb and move state ----------

    def learn_on_user(self, state: int, token_id: int) -> int:
        """
        Learn from user token (pure BU-Eg):
          - compute introns via ψ (big-endian bytes)
          - token_phase = fold(introns)
          - new_phase   = fold(rep_phase, token_phase)
          - register token in rep_channel[rep_cur][new_phase]
          - update rep_phase[rep_cur] = new_phase
          - step state by introns (egress)
          - passive diagnostics bound to canonical address (does not affect emission)
        """
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")

        idx = self.state_to_index[state]
        rep_cur = self.orbit_rep_index(idx)
        print(f"[DEBUG-LEARN] Learning state=0x{state:012X} idx={idx} rep_cur={rep_cur}")

        # Token intron micro-path and phase
        introns = token_to_introns(token_id)
        token_phase, _ = self.fold_sequence(introns, 0)

        # Compute the cumulative phase AFTER learning and register there
        cur_phase = self.rep_phase.get(rep_cur, 0)
        new_phase, _ = self._fold8(cur_phase, token_phase)

        # Get state_int for slab computation
        state_int = int(self.keys[idx])

        # Protected mutation section
        with self._lock:
            # === SLAB-SPECIFIC CHANNELS ===
            # Each slab gets its own phase and channel based on state geometry
            # CGM Theory: Store under cumulative context phase for proper chaining

            for slab_idx in range(8):
                # Context key: pure fold of cur_phase + geometry (NO TIME)
                slab_byte = self._slab_byte(state_int, slab_idx)
                ctx, _ = self._fold8(cur_phase, slab_byte)

                # Each slab maintains its own channel
                slab_chan = self.rep_channel.setdefault((rep_cur, slab_idx), {})
                bucket = slab_chan.setdefault(ctx, [])
                if token_id not in bucket:
                    print(f"[DEBUG-LEARN] Storing token {token_id} in rep={rep_cur} slab={slab_idx} ctx={ctx:02x}")
                    # Apply bucket capacity discipline with FIFO eviction
                    if len(bucket) >= self._max_bucket_size:
                        bucket.pop(0)  # Remove oldest token (FIFO)
                    bucket.append(token_id)

            # Store the updated per-orbit phase memory
            self.rep_phase[rep_cur] = new_phase

            # (passive diagnostics removed)

            # Mark changes pending and update counter
            self._pending_changes = True
            self._token_counter += 1

        # Step state by introns (egress)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)

        # Store content using the FINAL state's representation index
        final_rep_cur = self.orbit_rep_index(new_idx)
        final_state_int = int(self.keys[new_idx])
        
        # Re-store content in the final state's representation using same context computation
        for slab_idx in range(8):
            # Use the same context computation as the initial storage (pure context)
            slab_byte = self._slab_byte(final_state_int, slab_idx)
            ctx, _ = self._fold8(cur_phase, slab_byte)
            
            slab_chan = self.rep_channel.setdefault((final_rep_cur, slab_idx), {})
            bucket = slab_chan.setdefault(ctx, [])
            if token_id not in bucket:
                print(f"[DEBUG-LEARN] Re-storing token {token_id} in final rep={final_rep_cur} slab={slab_idx} ctx={ctx:02x}")
                if len(bucket) >= self._max_bucket_size:
                    bucket.pop(0)
                bucket.append(token_id)

        # Conditional persistence based on cadence
        self._maybe_save_learned_data()

        return int(self.keys[new_idx])

    def transit_on_assistant(self, state: int, token_id: int) -> int:
        """
        Assistant tokens: transit only; no learning.
        """
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        introns = token_to_introns(token_id)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)
        return int(self.keys[new_idx])

    def transit_on_control(self, state: int, token_id: int) -> int:
        return state

    # ---------- Ingress (BU-In): pure monodromic unfold ----------

    def emit_next(self, idx: int,
                  session_omega: Optional[Dict[Tuple[int,int], int]] = None,
                  session_bucket_key: Optional[Dict[Tuple[int,int], int]] = None,
                  session_bucket_pos: Optional[Dict[Tuple[int,int], Dict[int, int]]] = None,
                  session_monodromy: Optional[Dict[Tuple[int,int], int]] = None,
                  recent_egress_phases: Optional[List[int]] = None,
                  session_slab_cursor: Optional[Dict[int, int]] = None
) -> Optional[Tuple[int, int, Dict[Tuple[int,int], int], Dict[Tuple[int,int], int], Dict[Tuple[int,int], Dict[int, int]], Dict[Tuple[int,int], int], Dict[int,int]]]:
        """
        Pure gyro-walk: BU-In selects the next token by continuing the monodromic path.
        No ranking, no scores, no thresholds - just traceable walking.
        """
        rep_idx = self.orbit_rep_index(idx)
        state_int = int(self.keys[idx])

        omega = session_omega or {}
        bucket_key = session_bucket_key or {}
        bucket_pos = session_bucket_pos or {}
        monodromy = session_monodromy or {}
        slab_cursor = session_slab_cursor or {}

        # Endogenous stop: if alignment amplitude is 0 at entry, close path
        if self._alignment_amp(state_int, rep_idx) == 0:
            return None

        # Active slabs by sector; if none, use all
        sector_bits = self.sector(state_int)
        active_slabs = [s for s in range(8) if (sector_bits >> s) & 1]
        if not active_slabs:
            active_slabs = list(range(8))

        # round-robin order (no phase alignment / no scoring)
        start = slab_cursor.get(rep_idx, 0) % len(active_slabs)
        slab_order = [active_slabs[(start + hop) % len(active_slabs)] for hop in range(len(active_slabs))]

        # Live phases
        sp, sp_vel = self._state_phase(state_int)
        rp = self.rep_phase.get(rep_idx, 0)

        # Deterministic time tick (lawful variation)
        tick = self._free_tick()

        for slab_idx in slab_order:
            key = (rep_idx, slab_idx)
            slab_chan = self.rep_channel.get((rep_idx, slab_idx), {})
            if not slab_chan:
                continue

            keys = sorted(slab_chan.keys())
            n = len(keys)
            if n == 0:
                continue

            # Context addressing: use learning-time rule: ctx = fold(rp, slab_byte)
            slab_byte = self._slab_byte(state_int, slab_idx)
            ctx, _ = self._fold8(rp, slab_byte)

            # initialize rotor origin if first time
            if key not in bucket_key:
                # map ctx deterministically into ring
                bucket_key[key] = keys[ctx % n]
                if key not in bucket_pos:
                    bucket_pos[key] = {}

            import bisect
            base_val = bucket_key[key]
            base_idx = bisect.bisect_left(keys, base_val) % n

            # Ring stride derived from sp, omega, and tick — must be co-prime with n
            omega_val = omega.get(key, 0)
            stride_seed, _ = self._fold8(sp, omega_val ^ tick)
            stride = self._coprime_stride(stride_seed, n)

            # walk up to n steps to find a non-empty bucket
            for step_count in range(n):
                current_idx = (base_idx + step_count * stride) % n
                current_key = keys[current_idx]
                bucket = slab_chan.get(current_key, [])
                if not bucket:
                    continue

                # Intra-bucket rotor: step size from sp_vel and monodromy
                mono = monodromy.get(key, 0)
                inner_seed, _ = self._fold8(sp_vel, mono ^ tick)
                L = len(bucket)
                inner_stride = self._coprime_stride(inner_seed, L)

                pos_map = bucket_pos.setdefault(key, {})
                base_pos = pos_map.get(current_key, 0)

                tried = 0
                while tried < L:
                    pos = (base_pos + tried * inner_stride) % L
                    candidate = bucket[pos]

                    c_phase, _ = self.fold_sequence(token_to_introns(candidate), 0)

                    gate_acc = self.emit_gate.get(rep_idx, 0)
                    gated, _ = self._fold8(gate_acc, c_phase)

                    # core gate only (no scores/thresholds)
                    core_ok = True
                    if self.enable_core_gate:
                        last_tok = self.last_token.get(rep_idx, -1)
                        last_tick = self.last_emit_tick.get(rep_idx, tick)
                        dt = (tick - last_tick) & 0xFF
                        # egress mask: forbid recently emitted phases
                        if recent_egress_phases:
                            tphase, _ = self.token_phase(candidate)
                            if tphase in recent_egress_phases:
                                core_ok = False
                        # refractory: suppress immediate repeat under zero-fold with dt
                        if core_ok and candidate == last_tok and self._fold8(dt, c_phase)[0] == 0:
                            core_ok = False

                    ok = (gated != 0) and core_ok
                    if ok:
                        token_id = candidate
                        # Commit rotor state and traces
                        self.emit_gate[rep_idx] = gated
                        self.last_token[rep_idx] = token_id
                        self.last_emit_tick[rep_idx] = tick

                        # Omega and monodromy updates (path memory)
                        omega[key], _ = self._fold8((omega.get(key, 0) + 1) & 0xFF, c_phase)
                        monodromy[key], _ = self._fold8(mono, c_phase)

                        # Hop one ring location by stride for the next call
                        bucket_key[key] = keys[(current_idx + stride) % n]
                        pos_map[current_key] = (pos + 1) % L

                        # Advance canonical state by token introns (BU-In feeds back into state)
                        new_idx = idx
                        for i in token_to_introns(token_id):
                            new_idx = self.apply_intron_index(new_idx, i)

                        # Advance slab cursor round-robin (no "best slab")
                        slab_cursor[rep_idx] = (slab_cursor.get(rep_idx, 0) + 1) % max(1, len(active_slabs))

                        return token_id, new_idx, omega, bucket_key, bucket_pos, monodromy, slab_cursor

                    tried += 1

            # no token emitted from this slab; try next slab

        # No slabs emitted — endogenously signal no coherent continuation (caller decides stop/loop)
        return None

    # ---------- Harmony Integration Helpers ----------

    def is_harmony_control_token(self, token_id: int) -> bool:
        from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS

        return token_id in ALL_CONTROL_TOKENS

    def should_learn_from_token(self, token_id: int, role: str) -> bool:
        return role == "user" and not self.is_harmony_control_token(token_id) and 0 <= token_id < self.vocab_max

    def process_harmony_message(self, tokens: List[int], roles: List[str]) -> int:
        state = self.start_state()
        for token_id, role in zip(tokens, roles):
            if self.should_learn_from_token(token_id, role):
                state = self.learn_on_user(state, token_id)
        return self.state_to_index[state]

    # ---------- Interface Compatibility ----------

    def evolve_on_user(self, state: int, token_id: int) -> int:
        return self.learn_on_user(state, token_id)

    def evolve_on_assistant(self, state: int, token_id: int) -> int:
        return self.transit_on_assistant(state, token_id)

    def emit_next_from_state(self, state_int: int,
                            session_omega: Optional[Dict[Tuple[int,int], int]] = None,
                            session_bucket_key: Optional[Dict[Tuple[int,int], int]] = None,
                            session_bucket_pos: Optional[Dict[Tuple[int,int], Dict[int, int]]] = None,
                            session_monodromy: Optional[Dict[Tuple[int,int], int]] = None,
                            recent_egress_phases: Optional[List[int]] = None,
                            session_slab_cursor: Optional[Dict[int, int]] = None
) -> Optional[Tuple[int, int, Dict[Tuple[int,int], int], Dict[Tuple[int,int], int], Dict[Tuple[int,int], Dict[int, int]], Dict[Tuple[int,int], int], Dict[int,int]]]:
        if state_int not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_int:012X}")
        idx = self.state_to_index[state_int]
        res = self.emit_next(idx, session_omega, session_bucket_key, session_bucket_pos, session_monodromy, recent_egress_phases, session_slab_cursor)
        if res is None:
            return None
        tok, new_idx, omega, bucket_key, bucket_pos, monodromy, slab_cursor = res
        return tok, int(self.keys[new_idx]), omega, bucket_key, bucket_pos, monodromy, slab_cursor

    def next_token_aligned(self, state: int) -> Optional[int]:
        out = self.emit_next_from_state(state)
        return None if out is None else out[0]

    def next_token(self, state: int) -> Optional[int]:
        out = self.emit_next_from_state(state)
        return None if out is None else out[0]

    def start_state(self) -> int:
        return int(self.keys[self.start_index])

    def get_theta(self, state: int) -> float:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        return self.theta_of_index(idx)

    def get_orbit_representative(self, state: int) -> int:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        rep_idx = self.orbit_rep_index(idx)
        return int(self.keys[rep_idx])

    def get_orbit_size(self, state: int) -> int:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        return int(self.orbit_sizes[idx])

    def apply_intron(self, state: int, intron: int) -> int:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        new_idx = self.apply_intron_index(idx, intron)
        return int(self.keys[new_idx])

    def micro_path(self, start_state: int, introns: List[int]) -> List[int]:
        if start_state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{start_state:012X}")
        idx = self.state_to_index[start_state]
        path = [start_state]
        cur_idx = idx
        for intron in introns:
            cur_idx = self.apply_intron_index(cur_idx, intron)
            path.append(int(self.keys[cur_idx]))
        return path

    # ---------- Persistence ----------

    def _ensure_store_files(self):
        """Ensure that directories and files for persistence exist."""
        if not self.store_paths:
            return
        for path_str in self.store_paths.values():
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()

    def _load_learned_data(self):
        """Load learned phase and channel data from disk."""
        if not self.store_paths:
            return

        with self._lock:
            phase_path = self.store_paths.get("rep_phase")
            if phase_path and os.path.exists(phase_path) and os.path.getsize(phase_path) > 0:
                try:
                    with open(phase_path, "rb") as f:
                        self.rep_phase = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    self.rep_phase = {} # Start fresh on corruption

            channel_path = self.store_paths.get("rep_channel")
            if channel_path and os.path.exists(channel_path) and os.path.getsize(channel_path) > 0:
                try:
                    with open(channel_path, "rb") as f:
                        loaded_channel = pickle.load(f)

                    # Check if we need to migrate from old format to new slab-based format
                    if loaded_channel and isinstance(list(loaded_channel.keys())[0], int):
                        # Old format: Dict[int, Dict[int, List[int]]] - need to convert
                        print("Migrating rep_channel from old format to slab-based format...")
                        self.rep_channel = {}  # Start fresh with new format
                    else:
                        # New format: Dict[Tuple[int, int], Dict[int, List[int]]] - use as is
                        self.rep_channel = loaded_channel

                except (pickle.UnpicklingError, EOFError):
                    self.rep_channel = {} # Start fresh on corruption

    def _save_learned_data(self):
        """Save learned phase and channel data to disk."""
        if not self.store_paths:
            return

        with self._lock:
            if not self._pending_changes:
                return

            phase_path = self.store_paths.get("rep_phase")
            if phase_path:
                with open(phase_path, "wb") as f:
                    pickle.dump(self.rep_phase, f)

            channel_path = self.store_paths.get("rep_channel")
            if channel_path:
                with open(channel_path, "wb") as f:
                    pickle.dump(self.rep_channel, f)

            self._pending_changes = False
            self._last_save_time = time.time()
            self._token_counter = 0 # Reset counter after save

    def _maybe_save_learned_data(self):
        """Check if conditions are met to save learned data."""
        now = time.time()
        time_since_save = now - self._last_save_time

        should_save = (
            self._pending_changes and
            (self._token_counter >= self._save_interval_tokens or
             time_since_save >= self._save_interval_seconds)
        )

        if should_save:
            self._save_learned_data()

    # ---------- Toroidal Routing ----------

    def sector(self, state_int: int) -> int:
        """
        Compute 8-bit toroidal signature from 48-bit state using proper slab parities.
        Uses frozen slab structure with correct bit indices from FROZEN_CHANNELS.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS

        sector_bits = 0
        for slab_idx in range(FROZEN_CHANNELS.NUM_SLABS):  # 8 slabs
            # Calculate the parity for the current slab
            parity = 0
            for bit_idx in FROZEN_CHANNELS.get_slab_bit_indices(slab_idx):
                if (state_int >> bit_idx) & 1:
                    parity ^= 1

            # Set the corresponding bit in the sector signature
            if parity:
                sector_bits |= (1 << slab_idx)

        return sector_bits

