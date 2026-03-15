from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

import src.api as api_mod
import src.constants as constants_mod
from src.api import (
    EPS_A6_BY_BYTE,
    EPS_B6_BY_BYTE,
    MICRO_REF_BY_BYTE,
    omega_word_signature,
    omega12_to_state24,
    pack_omega_signature12,
    q_word6,
    shell_krawtchouk_transform_exact,
    state24_to_omega12,
    trajectory_parity_commitment,
    unpack_omega12,
    walsh_hadamard64,
)
from src.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    ab_distance,
    byte_family,
    byte_micro_ref,
    horizon_distance,
    unpack_state,
)
from src.sdk import (
    RuntimeOps,
    future_cone_measure,
    future_locus_measure,
    initialize_native,
    optical_coordinates,
    state_charts,
    stabilizer_type,
)
from src.tools.gyrolabe import ops as gyrolabe_ops
from . import ops as gg_ops
from .profiles import (
    ResonanceProfile,
    bucket_count,
    chi6_from_omega12,
    key_for_closed_word,
    shell_from_omega12,
)
from .serializers import ensure_word4

_MAGIC = b"GYRG"
_VERSION = 1

# magic, version, capacity, active_count, profile_id, flags,
# created_unix_ns, kernel_hash32
_HEADER = struct.Struct("<4sIIIHHQ32s")

# uint32 cell_id + raw 4-byte word
_INGEST_REC = struct.Struct("<I4s")

_ZERO_WORD4 = b"\x00\x00\x00\x00"

_rest_omega = state24_to_omega12(GENE_MAC_REST)
_REST_OMEGA12 = ((_rest_omega.u6 & 0x3F) << 6) | (_rest_omega.v6 & 0x3F)


def step_packed_omega12(omega12: int, byte: int) -> int:
    """
    Packed Ω step using the existing api.py Ω tables.

    This is algebraically identical to src.api.step_omega12_by_byte,
    but works directly on packed omega12 ints.
    """
    x = int(omega12) & 0xFFF
    b = int(byte) & 0xFF
    u6 = (x >> 6) & 0x3F
    v6 = x & 0x3F
    u_next = v6 ^ EPS_A6_BY_BYTE[b]
    v_next = u6 ^ MICRO_REF_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b]
    return ((u_next & 0x3F) << 6) | (v_next & 0x3F)


def _state24_from_packed_omega12(omega12: int) -> int:
    return omega12_to_state24(unpack_omega12(int(omega12) & 0xFFF))


def _kernel_law_hash() -> bytes:
    """
    Compute a stable law hash from the core kernel-law surfaces.

    This intentionally covers the Python and native law-carrying files
    that define the exact stepping behavior.
    """
    h = hashlib.sha256()

    paths: list[Path] = []
    for mod in (constants_mod, api_mod):
        p = getattr(mod, "__file__", None)
        if p is not None:
            paths.append(Path(p))

    try:
        gyrolabe_dir = Path(__file__).resolve().parents[1] / "gyrolabe"
        for name in ("gyrolabe.c", "gyrolabe_opencl.c"):
            p = gyrolabe_dir / name
            if p.exists():
                paths.append(p)
    except Exception:
        pass

    for p in paths:
        if p.exists():
            h.update(p.read_bytes())

    return h.digest()


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data


def _read_array(f, dtype, shape) -> np.ndarray:
    dt = np.dtype(dtype)
    count = int(np.prod(shape))
    raw = _read_exact(f, count * dt.itemsize)
    arr = np.frombuffer(raw, dtype=dt, count=count).copy()
    return arr.reshape(shape)


@dataclass(frozen=True)
class SLCPRecord:
    cell_id: int
    step: int
    omega12: int
    state24: int
    last_byte: int
    family: int
    micro_ref: int
    q6: int
    chi6: int
    shell: int
    horizon_distance: int
    ab_distance: int
    omega_sig: int
    parity_O12: int
    parity_E12: int
    parity_bit: int
    resonance_key: int
    current_resonance: int
    spectral64: np.ndarray

    def charts(self):
        return state_charts(self.state24)

    def future_cone(self, n: int):
        return future_cone_measure(self.state24, n)

    def future_locus(self, n: int):
        return future_locus_measure(self.state24, n)

    def optical_coordinates(self):
        return optical_coordinates(self.state24)

    def stabilizer_type(self):
        return stabilizer_type(self.state24)


class GyroGraph:
    """
    Ω-native multicellular runtime machine.

    Core state is stored in structure-of-arrays layout.
    Local structural memory updates at byte cadence.
    Resonance updates at word closure.
    """

    def __init__(
        self,
        cell_capacity: int,
        profile: ResonanceProfile = ResonanceProfile.CHIRALITY,
        *,
        enable_ingest_log: bool = False,
        ingest_log_path: str | None = None,
        use_native_hotpath: bool = True,
        use_opencl_hotpath: bool = True,
        opencl_min_batch: int = 128,
        opencl_platform_index: int = 0,
        opencl_device_index: int = 0,
    ) -> None:
        initialize_native()

        self._use_native_hotpath = bool(use_native_hotpath and gg_ops.native_available())
        self._use_opencl_hotpath = bool(use_opencl_hotpath and gg_ops.opencl_available())
        self._opencl_min_batch = max(1, int(opencl_min_batch))

        if self._use_opencl_hotpath:
            try:
                gg_ops.initialize_opencl(
                    int(opencl_platform_index),
                    int(opencl_device_index),
                )
            except Exception:
                self._use_opencl_hotpath = False

        if int(cell_capacity) <= 0:
            raise ValueError("cell_capacity must be positive")

        self._profile = ResonanceProfile(profile)
        self._enable_ingest_log = bool(enable_ingest_log)
        self._created_unix_ns = time.time_ns()

        self._ingest_log_path: Path | None = None
        self._ingest_log_records: list[tuple[int, bytes]] | None = (
            [] if self._enable_ingest_log else None
        )

        self._allocate_storage(int(cell_capacity))

        if ingest_log_path is not None:
            self.set_ingest_log_path(ingest_log_path)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _allocate_storage(self, capacity: int) -> None:
        n = int(capacity)
        self._capacity = n

        self._allocated = np.zeros(n, dtype=np.bool_)
        self._has_closed_word = np.zeros(n, dtype=np.bool_)

        # Core state
        self._omega12 = np.zeros(n, dtype=np.int32)
        self._step = np.zeros(n, dtype=np.uint64)
        self._last_byte = np.full(n, GENE_MIC_S, dtype=np.uint8)

        # Most recent closed word (zeros until first word closure)
        self._word4 = np.zeros((n, 4), dtype=np.uint8)

        # Rolling chirality memory
        self._chi_ring64 = np.zeros((n, 64), dtype=np.uint8)
        self._chi_ring_pos = np.zeros(n, dtype=np.uint8)
        self._chi_valid_len = np.zeros(n, dtype=np.uint8)

        # Rolling distributions
        self._chi_hist64 = np.zeros((n, 64), dtype=np.uint16)
        self._shell_hist7 = np.zeros((n, 7), dtype=np.uint16)

        # Most recent compiled word action
        self._omega_sig = np.zeros(n, dtype=np.int32)

        # Most recent parity commitment of the stored closed word
        self._parity_O12 = np.zeros(n, dtype=np.uint16)
        self._parity_E12 = np.zeros(n, dtype=np.uint16)
        self._parity_bit = np.zeros(n, dtype=np.uint8)

        # Resonance
        self._resonance_key = np.zeros(n, dtype=np.uint32)
        self._resonance_buckets = np.zeros(
            bucket_count(self._profile), dtype=np.uint64
        )
        self._bucket_cells: list[set[int]] = [
            set() for _ in range(self._resonance_buckets.size)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_allocated(self, cell_id: int) -> None:
        cid = int(cell_id)
        if cid < 0 or cid >= self._capacity:
            raise ValueError(
                f"cell_id {cid} out of range [0, {self._capacity})"
            )
        if not bool(self._allocated[cid]):
            raise ValueError(f"cell {cid} is not allocated")

    def _check_bucket_key(self, key: int) -> int:
        k = int(key)
        if k < 0 or k >= self._resonance_buckets.size:
            raise ValueError(
                f"bucket key {k} out of range [0, {self._resonance_buckets.size})"
            )
        return k

    def _clear_cell_memory(self, cell_id: int, omega12: int) -> None:
        cid = int(cell_id)

        self._omega12[cid] = int(omega12) & 0xFFF
        self._step[cid] = 0
        self._last_byte[cid] = GENE_MIC_S

        self._word4[cid, :] = 0
        self._has_closed_word[cid] = False

        self._chi_ring64[cid, :] = 0
        self._chi_ring_pos[cid] = 0
        self._chi_valid_len[cid] = 0

        self._chi_hist64[cid, :] = 0
        self._shell_hist7[cid, :] = 0

        self._omega_sig[cid] = 0
        self._parity_O12[cid] = 0
        self._parity_E12[cid] = 0
        self._parity_bit[cid] = 0

        self._resonance_key[cid] = 0

    def _install_resonance_key(self, cell_id: int, key: int) -> None:
        cid = int(cell_id)
        k = self._check_bucket_key(key)
        self._resonance_key[cid] = k
        self._bucket_cells[k].add(cid)
        self._resonance_buckets[k] = len(self._bucket_cells[k])

    def _remove_from_resonance(self, cell_id: int) -> None:
        cid = int(cell_id)
        k = self._check_bucket_key(int(self._resonance_key[cid]))
        self._bucket_cells[k].discard(cid)
        self._resonance_buckets[k] = len(self._bucket_cells[k])

    def _move_resonance(self, cell_id: int, new_key: int) -> None:
        cid = int(cell_id)
        old_key = self._check_bucket_key(int(self._resonance_key[cid]))
        new_key_i = self._check_bucket_key(int(new_key))
        if old_key == new_key_i:
            return
        self._bucket_cells[old_key].discard(cid)
        self._resonance_buckets[old_key] = len(self._bucket_cells[old_key])
        self._bucket_cells[new_key_i].add(cid)
        self._resonance_buckets[new_key_i] = len(self._bucket_cells[new_key_i])
        self._resonance_key[cid] = new_key_i

    def decay_resonance_buckets(self) -> None:
        """
        Decay resonance bucket weights by one bit (halving).
        Does not change cell membership; SLCP current_resonance will report halved values.
        """
        self._resonance_buckets >>= 1

    def _push_chi(self, cell_id: int, chi6: int) -> None:
        cid = int(cell_id)
        chi = int(chi6) & 0x3F
        shell = chi.bit_count()

        pos = int(self._chi_ring_pos[cid])
        valid = int(self._chi_valid_len[cid])

        if valid < 64:
            self._chi_ring64[cid, pos] = chi
            self._chi_hist64[cid, chi] += 1
            self._shell_hist7[cid, shell] += 1
            self._chi_ring_pos[cid] = (pos + 1) & 63
            self._chi_valid_len[cid] = valid + 1
            return

        chi_old = int(self._chi_ring64[cid, pos])
        shell_old = chi_old.bit_count()

        self._chi_hist64[cid, chi_old] -= 1
        self._shell_hist7[cid, shell_old] -= 1

        self._chi_ring64[cid, pos] = chi
        self._chi_hist64[cid, chi] += 1
        self._shell_hist7[cid, shell] += 1
        self._chi_ring_pos[cid] = (pos + 1) & 63

    def _parse_packet(
        self,
        packet,
    ) -> tuple[int, bytes]:
        if not isinstance(packet, tuple):
            raise TypeError(
                "Each packet must be a tuple of (cell_id, word4) or "
                "(cell_id, word4, bridge_metadata)"
            )
        if len(packet) == 2:
            cell_id, word4 = packet
            return int(cell_id), ensure_word4(word4)
        if len(packet) == 3:
            cell_id, word4, _bridge_metadata = packet
            return int(cell_id), ensure_word4(word4)
        raise ValueError(
            "Each packet must be length 2 or 3: "
            "(cell_id, word4) or (cell_id, word4, bridge_metadata)"
        )

    def _spectral64(self, cell_id: int) -> np.ndarray:
        cid = int(cell_id)
        x = self._chi_hist64[cid].astype(np.float32, copy=False)

        try:
            import torch
            from src.tools.gyrolabe import ops as gyrolabe_ops

            x_t = torch.from_numpy(x).to(dtype=torch.float32).reshape(1, 64)
            y_t = gyrolabe_ops.wht64(x_t)
            y = y_t.reshape(64).detach().cpu().numpy()
            return y.astype(np.float32, copy=False)
        except Exception:
            W = walsh_hadamard64()
            y = W @ x.astype(np.float64)
            return y.astype(np.float32, copy=False)

    def _append_ingest_record(self, cell_id: int, word4: bytes) -> None:
        if not self._enable_ingest_log:
            return

        if self._ingest_log_records is not None:
            self._ingest_log_records.append((int(cell_id), bytes(word4)))

        if self._ingest_log_path is None:
            return

        self._ingest_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._ingest_log_path, "ab") as f:
            f.write(_INGEST_REC.pack(int(cell_id), ensure_word4(word4)))

    def _gather_native_state(self, ids: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "omega12": np.ascontiguousarray(self._omega12[ids], dtype=np.int32),
            "step": np.ascontiguousarray(self._step[ids], dtype=np.uint64),
            "last_byte": np.ascontiguousarray(self._last_byte[ids], dtype=np.uint8),
            "has_closed_word": np.ascontiguousarray(
                self._has_closed_word[ids], dtype=np.uint8
            ),
            "word4": np.ascontiguousarray(self._word4[ids], dtype=np.uint8),
            "chi_ring64": np.ascontiguousarray(self._chi_ring64[ids], dtype=np.uint8),
            "chi_ring_pos": np.ascontiguousarray(
                self._chi_ring_pos[ids], dtype=np.uint8
            ),
            "chi_valid_len": np.ascontiguousarray(
                self._chi_valid_len[ids], dtype=np.uint8
            ),
            "chi_hist64": np.ascontiguousarray(self._chi_hist64[ids], dtype=np.uint16),
            "shell_hist7": np.ascontiguousarray(self._shell_hist7[ids], dtype=np.uint16),
            "omega_sig": np.ascontiguousarray(self._omega_sig[ids], dtype=np.int32),
            "parity_O12": np.ascontiguousarray(self._parity_O12[ids], dtype=np.uint16),
            "parity_E12": np.ascontiguousarray(self._parity_E12[ids], dtype=np.uint16),
            "parity_bit": np.ascontiguousarray(self._parity_bit[ids], dtype=np.uint8),
        }

    def _scatter_native_state(
        self, ids: np.ndarray, state: dict[str, np.ndarray]
    ) -> None:
        self._omega12[ids] = state["omega12"]
        self._step[ids] = state["step"]
        self._last_byte[ids] = state["last_byte"]
        self._has_closed_word[ids] = state["has_closed_word"].astype(np.bool_)
        self._word4[ids] = state["word4"]
        self._chi_ring64[ids] = state["chi_ring64"]
        self._chi_ring_pos[ids] = state["chi_ring_pos"]
        self._chi_valid_len[ids] = state["chi_valid_len"]
        self._chi_hist64[ids] = state["chi_hist64"]
        self._shell_hist7[ids] = state["shell_hist7"]
        self._omega_sig[ids] = state["omega_sig"]
        self._parity_O12[ids] = state["parity_O12"]
        self._parity_E12[ids] = state["parity_E12"]
        self._parity_bit[ids] = state["parity_bit"]

    def _ingest_batch_native(self, parsed: list[tuple[int, bytes]]) -> None:
        ids = np.asarray([cid for cid, _ in parsed], dtype=np.int64)
        words4 = np.vstack(
            [np.frombuffer(word4, dtype=np.uint8) for _, word4 in parsed]
        ).astype(np.uint8, copy=False)

        state = self._gather_native_state(ids)

        if self._use_opencl_hotpath and len(parsed) >= self._opencl_min_batch:
            omega_trace4, chi_trace4 = gg_ops.trace_word4_batch_opencl(
                state["omega12"],
                words4,
            )
            gg_ops.apply_trace_word4_batch(
                state["omega12"],
                state["step"],
                state["last_byte"],
                state["has_closed_word"],
                state["word4"],
                state["chi_ring64"],
                state["chi_ring_pos"],
                state["chi_valid_len"],
                state["chi_hist64"],
                state["shell_hist7"],
                state["omega_sig"],
                state["parity_O12"],
                state["parity_E12"],
                state["parity_bit"],
                words4,
                omega_trace4,
                chi_trace4,
            )
        else:
            gg_ops.ingest_word4_batch(
                state["omega12"],
                state["step"],
                state["last_byte"],
                state["has_closed_word"],
                state["word4"],
                state["chi_ring64"],
                state["chi_ring_pos"],
                state["chi_valid_len"],
                state["chi_hist64"],
                state["shell_hist7"],
                state["omega_sig"],
                state["parity_O12"],
                state["parity_E12"],
                state["parity_bit"],
                words4,
            )

        self._scatter_native_state(ids, state)

        for row, cid in enumerate(ids.tolist()):
            word4 = bytes(words4[row].tolist())
            new_key = key_for_closed_word(
                self._profile,
                omega12=int(self._omega12[cid]),
                word4=word4,
                omega_sig=int(self._omega_sig[cid]),
            )
            self._move_resonance(int(cid), new_key)
            self._append_ingest_record(int(cid), word4)

    def _rebuild_bucket_cells_from_state(self) -> None:
        self._bucket_cells = [set() for _ in range(self._resonance_buckets.size)]
        counts = np.zeros_like(self._resonance_buckets)
        for cid in np.flatnonzero(self._allocated):
            k = self._check_bucket_key(int(self._resonance_key[cid]))
            self._bucket_cells[k].add(int(cid))
            counts[k] += 1

        if not np.array_equal(counts, self._resonance_buckets):
            raise ValueError(
                "Snapshot resonance bucket counts do not match stored resonance keys"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_cells(self, count: int = 1) -> list[int]:
        count_i = int(count)
        if count_i <= 0:
            raise ValueError("count must be positive")

        free_ids = np.flatnonzero(~self._allocated)
        if free_ids.size < count_i:
            raise RuntimeError("Not enough free cells available")

        out = free_ids[:count_i].tolist()
        for cid in out:
            self._allocated[cid] = True
            self._clear_cell_memory(cid, _REST_OMEGA12)
            key = key_for_closed_word(
                self._profile,
                omega12=int(self._omega12[cid]),
                word4=_ZERO_WORD4,
                omega_sig=0,
            )
            self._install_resonance_key(cid, key)

        return out

    def free_cells(self, cell_ids: list[int]) -> None:
        for cid in cell_ids:
            self._check_allocated(cid)
            self._remove_from_resonance(cid)
            self._allocated[cid] = False
            self._clear_cell_memory(cid, 0)

    def seed_rest(self, cell_ids: list[int]) -> None:
        for cid in cell_ids:
            self._check_allocated(cid)
            self._remove_from_resonance(cid)
            self._clear_cell_memory(cid, _REST_OMEGA12)
            key = key_for_closed_word(
                self._profile,
                omega12=int(self._omega12[cid]),
                word4=_ZERO_WORD4,
                omega_sig=0,
            )
            self._install_resonance_key(cid, key)

    def seed_equality_horizon(self, cell_ids: list[int]) -> None:
        """
        Deterministic equality-horizon representative:
        packed omega12 with u6=0, v6=0.
        """
        eq_omega12 = 0
        for cid in cell_ids:
            self._check_allocated(cid)
            self._remove_from_resonance(cid)
            self._clear_cell_memory(cid, eq_omega12)
            key = key_for_closed_word(
                self._profile,
                omega12=int(self._omega12[cid]),
                word4=_ZERO_WORD4,
                omega_sig=0,
            )
            self._install_resonance_key(cid, key)

    def seed_shell(self, cell_ids: list[int], shell: int) -> None:
        s = int(shell)
        if not (0 <= s <= 6):
            raise ValueError("shell must be in 0..6")

        chi6 = 0
        for i in range(s):
            chi6 |= 1 << i

        # Deterministic shell representative: u6=0, v6=chi6
        omega12 = chi6 & 0x3F

        for cid in cell_ids:
            self._check_allocated(cid)
            self._remove_from_resonance(cid)
            self._clear_cell_memory(cid, omega12)
            key = key_for_closed_word(
                self._profile,
                omega12=int(self._omega12[cid]),
                word4=_ZERO_WORD4,
                omega_sig=0,
            )
            self._install_resonance_key(cid, key)

    def seed_omega(self, cell_id: int, omega12: int) -> None:
        cid = int(cell_id)
        self._check_allocated(cid)
        self._remove_from_resonance(cid)
        self._clear_cell_memory(cid, int(omega12) & 0xFFF)
        key = key_for_closed_word(
            self._profile,
            omega12=int(self._omega12[cid]),
            word4=_ZERO_WORD4,
            omega_sig=0,
        )
        self._install_resonance_key(cid, key)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, packets: Iterable[tuple]) -> None:
        parsed = [self._parse_packet(packet) for packet in packets]
        if not parsed:
            return

        for cid, _ in parsed:
            self._check_allocated(cid)

        ids_in_batch = [cid for cid, _ in parsed]
        use_batch = (
            self._use_native_hotpath
            and len(parsed) > 1
            and len(ids_in_batch) == len(set(ids_in_batch))
        )
        if use_batch:
            self._ingest_batch_native(parsed)
            return

        for cid, word4 in parsed:
            self._ingest_word(cid, word4)

    def _ingest_word(self, cell_id: int, word4: bytes) -> None:
        cid = int(cell_id)
        self._check_allocated(cid)

        w = ensure_word4(word4)
        omega12 = int(self._omega12[cid])

        # Byte-cadence local memory update
        for b in w:
            omega12 = step_packed_omega12(omega12, b)
            self._step[cid] += 1
            self._last_byte[cid] = b
            self._push_chi(cid, chi6_from_omega12(omega12))

        # Closure-boundary state
        self._omega12[cid] = omega12
        self._word4[cid, :] = np.frombuffer(w, dtype=np.uint8)

        omega_sig = pack_omega_signature12(omega_word_signature(w))
        self._omega_sig[cid] = omega_sig

        O12, E12, pbit = trajectory_parity_commitment(w)
        self._parity_O12[cid] = O12
        self._parity_E12[cid] = E12
        self._parity_bit[cid] = pbit

        new_key = key_for_closed_word(
            self._profile,
            omega12=omega12,
            word4=w,
            omega_sig=omega_sig,
        )
        self._move_resonance(cid, new_key)
        self._has_closed_word[cid] = True

        self._append_ingest_record(cid, w)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit_slcp(self, cell_ids: list[int]) -> list[SLCPRecord]:
        if not cell_ids:
            return []

        ids = np.asarray(cell_ids, dtype=np.int64)
        for cid in ids.tolist():
            self._check_allocated(int(cid))

        omega_np = np.ascontiguousarray(self._omega12[ids], dtype=np.int32)
        last_np = np.ascontiguousarray(self._last_byte[ids], dtype=np.uint8)
        step_np = np.ascontiguousarray(self._step[ids], dtype=np.uint64)
        has_word_np = np.ascontiguousarray(
            self._has_closed_word[ids], dtype=np.bool_
        )
        omega_sig_np = np.ascontiguousarray(self._omega_sig[ids], dtype=np.int32)
        parity_O_np = np.ascontiguousarray(self._parity_O12[ids], dtype=np.uint16)
        parity_E_np = np.ascontiguousarray(self._parity_E12[ids], dtype=np.uint16)
        parity_bit_np = np.ascontiguousarray(
            self._parity_bit[ids], dtype=np.uint8
        )
        resonance_key_np = np.ascontiguousarray(
            self._resonance_key[ids], dtype=np.uint32
        )
        chi_hist_np = np.ascontiguousarray(
            self._chi_hist64[ids], dtype=np.float32
        )

        try:
            state_t = RuntimeOps.states_from_omega12(torch.from_numpy(omega_np))
            state_np = state_t.detach().cpu().numpy().astype(np.int32, copy=False)
        except Exception:
            state_np = np.asarray(
                [_state24_from_packed_omega12(int(x)) for x in omega_np],
                dtype=np.int32,
            )

        try:
            spectral_t = gyrolabe_ops.wht64(torch.from_numpy(chi_hist_np))
            spectral_np = spectral_t.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        except Exception:
            W = walsh_hadamard64()
            spectral_np = (chi_hist_np.astype(np.float64) @ W.T).astype(
                np.float32
            )

        chi_np = (((omega_np >> 6) ^ omega_np) & 0x3F).astype(np.uint8)
        shell_np = np.asarray(
            [int(x).bit_count() for x in chi_np.tolist()],
            dtype=np.uint8,
        )

        out: list[SLCPRecord] = []
        for j, cid in enumerate(ids.tolist()):
            state24 = int(state_np[j])
            a12, b12 = unpack_state(state24)

            resonance_key = int(resonance_key_np[j])
            current_resonance = int(self._resonance_buckets[resonance_key])

            has_word = bool(has_word_np[j])

            out.append(
                SLCPRecord(
                    cell_id=int(cid),
                    step=int(step_np[j]),
                    omega12=int(omega_np[j]),
                    state24=state24,
                    last_byte=int(last_np[j]),
                    family=byte_family(int(last_np[j])),
                    micro_ref=byte_micro_ref(int(last_np[j])),
                    q6=q_word6(int(last_np[j])),
                    chi6=int(chi_np[j]),
                    shell=int(shell_np[j]),
                    horizon_distance=horizon_distance(a12, b12),
                    ab_distance=ab_distance(a12, b12),
                    omega_sig=int(omega_sig_np[j]) if has_word else 0,
                    parity_O12=int(parity_O_np[j]) if has_word else 0,
                    parity_E12=int(parity_E_np[j]) if has_word else 0,
                    parity_bit=int(parity_bit_np[j]) if has_word else 0,
                    resonance_key=resonance_key,
                    current_resonance=current_resonance,
                    spectral64=spectral_np[j].copy(),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Graph query surface
    # ------------------------------------------------------------------

    def get_co_resonant_cells(self, cell_id: int) -> list[int]:
        self._check_allocated(cell_id)
        key = int(self._resonance_key[cell_id])
        return sorted(
            cid for cid in self._bucket_cells[key] if cid != int(cell_id)
        )

    def get_bucket_population(self, key: int) -> int:
        k = self._check_bucket_key(key)
        return int(self._resonance_buckets[k])

    def get_bucket_cells(self, key: int) -> list[int]:
        k = self._check_bucket_key(key)
        return sorted(self._bucket_cells[k])

    def get_cells_on_shell(self, shell: int) -> list[int]:
        s = int(shell)
        if not (0 <= s <= 6):
            raise ValueError("shell must be in 0..6")

        out: list[int] = []
        for cid in np.flatnonzero(self._allocated):
            if shell_from_omega12(int(self._omega12[cid])) == s:
                out.append(int(cid))
        return out

    def get_cells_with_chi6(self, chi6: int) -> list[int]:
        target = int(chi6) & 0x3F

        out: list[int] = []
        for cid in np.flatnonzero(self._allocated):
            if chi6_from_omega12(int(self._omega12[cid])) == target:
                out.append(int(cid))
        return out

    def get_cells_with_signature(self, omega_sig: int) -> list[int]:
        target = int(omega_sig) & 0x1FFF

        out: list[int] = []
        for cid in np.flatnonzero(self._allocated):
            if int(self._omega_sig[cid]) == target:
                out.append(int(cid))
        return out

    def chirality_distance_between_cells(self, a: int, b: int) -> int:
        self._check_allocated(a)
        self._check_allocated(b)
        chi_a = chi6_from_omega12(int(self._omega12[a]))
        chi_b = chi6_from_omega12(int(self._omega12[b]))
        return (chi_a ^ chi_b).bit_count()

    def shell_spectral(self, cell_id: int):
        self._check_allocated(cell_id)
        return shell_krawtchouk_transform_exact(
            [int(x) for x in self._shell_hist7[int(cell_id)]]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        flags = 0
        if self._enable_ingest_log:
            flags |= 0x1

        khash = _kernel_law_hash()

        with open(p, "wb") as f:
            f.write(
                _HEADER.pack(
                    _MAGIC,
                    _VERSION,
                    int(self._capacity),
                    int(self.active_cell_count),
                    int(self._profile),
                    int(flags),
                    int(self._created_unix_ns),
                    khash,
                )
            )

            f.write(self._allocated.tobytes(order="C"))
            f.write(self._has_closed_word.tobytes(order="C"))
            f.write(self._omega12.tobytes(order="C"))
            f.write(self._step.tobytes(order="C"))
            f.write(self._last_byte.tobytes(order="C"))
            f.write(self._word4.tobytes(order="C"))
            f.write(self._chi_ring64.tobytes(order="C"))
            f.write(self._chi_ring_pos.tobytes(order="C"))
            f.write(self._chi_valid_len.tobytes(order="C"))
            f.write(self._chi_hist64.tobytes(order="C"))
            f.write(self._shell_hist7.tobytes(order="C"))
            f.write(self._omega_sig.tobytes(order="C"))
            f.write(self._parity_O12.tobytes(order="C"))
            f.write(self._parity_E12.tobytes(order="C"))
            f.write(self._parity_bit.tobytes(order="C"))
            f.write(self._resonance_key.tobytes(order="C"))
            f.write(self._resonance_buckets.tobytes(order="C"))

    def restore(self, path: str) -> None:
        p = Path(path)
        with open(p, "rb") as f:
            header = _read_exact(f, _HEADER.size)
            (
                magic,
                version,
                capacity,
                _active_count,
                profile_id,
                flags,
                created_unix_ns,
                file_hash,
            ) = _HEADER.unpack(header)

            if magic != _MAGIC:
                raise ValueError(f"Invalid GyroGraph snapshot magic: {magic!r}")
            if version != _VERSION:
                raise ValueError(
                    f"Unsupported GyroGraph snapshot version: {version}"
                )

            current_hash = _kernel_law_hash()
            if file_hash != current_hash:
                raise ValueError(
                    "Kernel law hash mismatch between snapshot and current repository state"
                )

            self._profile = ResonanceProfile(profile_id)
            self._enable_ingest_log = bool(flags & 0x1)
            self._created_unix_ns = int(created_unix_ns)

            self._allocate_storage(int(capacity))

            self._allocated = _read_array(f, np.bool_, (self._capacity,))
            self._has_closed_word = _read_array(
                f, np.bool_, (self._capacity,)
            )
            self._omega12 = _read_array(f, np.int32, (self._capacity,))
            self._step = _read_array(f, np.uint64, (self._capacity,))
            self._last_byte = _read_array(f, np.uint8, (self._capacity,))
            self._word4 = _read_array(f, np.uint8, (self._capacity, 4))
            self._chi_ring64 = _read_array(f, np.uint8, (self._capacity, 64))
            self._chi_ring_pos = _read_array(f, np.uint8, (self._capacity,))
            self._chi_valid_len = _read_array(f, np.uint8, (self._capacity,))
            self._chi_hist64 = _read_array(f, np.uint16, (self._capacity, 64))
            self._shell_hist7 = _read_array(f, np.uint16, (self._capacity, 7))
            self._omega_sig = _read_array(f, np.int32, (self._capacity,))
            self._parity_O12 = _read_array(f, np.uint16, (self._capacity,))
            self._parity_E12 = _read_array(f, np.uint16, (self._capacity,))
            self._parity_bit = _read_array(f, np.uint8, (self._capacity,))
            self._resonance_key = _read_array(f, np.uint32, (self._capacity,))
            self._resonance_buckets = _read_array(
                f, np.uint64, (bucket_count(self._profile),)
            )

        self._rebuild_bucket_cells_from_state()

        # Ingest log is maintained separately.
        self._ingest_log_records = [] if self._enable_ingest_log else None
        self._ingest_log_path = None

    # ------------------------------------------------------------------
    # Ingest log support
    # ------------------------------------------------------------------

    def set_ingest_log_path(self, path: str) -> None:
        self._ingest_log_path = Path(path)

    def iter_ingest_log(self) -> tuple[tuple[int, bytes], ...]:
        if self._ingest_log_records is None:
            return ()
        return tuple(self._ingest_log_records)

    def clear_ingest_log(self) -> None:
        if self._ingest_log_records is not None:
            self._ingest_log_records.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return int(self._capacity)

    @property
    def profile(self) -> ResonanceProfile:
        return self._profile

    @property
    def active_cell_count(self) -> int:
        return int(np.count_nonzero(self._allocated))

    @property
    def active_cell_ids(self) -> list[int]:
        return [int(x) for x in np.flatnonzero(self._allocated)]

    @property
    def native_hotpath_enabled(self) -> bool:
        return bool(self._use_native_hotpath)

    @property
    def opencl_hotpath_enabled(self) -> bool:
        return bool(self._use_opencl_hotpath)

    @property
    def ingest_log_enabled(self) -> bool:
        return bool(self._enable_ingest_log)