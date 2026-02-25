"""
src.tools.layers
================

Multi-resolution agent memory substrate (NEW), independent from the legacy Router atlas.

IMPORTANT
---------
These layers:
  - DO NOT use ontology.npy / epistemology.npy / phenomenology.npz
  - DO NOT use RouterKernel at runtime
  - DO reuse FIRST-PRINCIPLES physics primitives/constants from src.router.constants
    (GENE_MIC_S, mask12_for_byte, pack/unpack, etc.)

Design
------
L1/L2/L3 are true finite-state machines (FSMs) with explicit transition tables:

  L1:  8-bit state   -> 256 states × 256 bytes = 65,536 transitions
  L2: 16-bit state   -> 65,536 states × 256    = 16,777,216 transitions
  L3: 24-bit state   -> 16,777,216 × 256       = 4,294,967,296 transitions

L4 is NOT a table. It is a dual operator over the stream:
  - maintains a tiny holographic register (O/E parity commitments)
  - provides closure/identity-cycle signals
  - provides exact inverse stepping for L3

Table semantics
--------------
The FSM tables store next_state = table[state, byte]. Transitions are "parameters".

Function face
-------------
Each FSM also has a function face (fixed-width bit ops). Tables are caches/artifacts
that can be verified against the function face.

Storage
-------
- L1 table: uint8[256,256]      = 64 KB
- L2 table: uint16[65536,256]   = 32 MB
- L3 table: packed uint24 next states:
    uint8[2^24, 256, 3] = 12.9 GB  (disk-backed memmap recommended)
  (uint32 form would be ~16.8 GB; packed is preferred)

Performance notes
-----------------
- L2 build is vectorized (fast).
- L3 build is chunked + vectorized per byte; still heavy (writes 12.9 GB).
  Use only if you truly want the table. Otherwise use function face for L3.

No CGM naming here; only L1/L2/L3/L4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# First-principles physics primitives/constants are allowed.
# (But: no atlas artifacts, no RouterKernel.)
from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_STATE24,
    GENE_MIC_S,
    LAYER_MASK_12,
    mask12_for_byte,
    pack_state,
    popcount,
    unpack_state,
)

# =============================================================================
# Shared primitives
# =============================================================================


def intron(byte_val: int) -> int:
    """Transcription: intron = byte XOR 0xAA."""
    return (int(byte_val) & 0xFF) ^ GENE_MIC_S


def _u24(x: int) -> int:
    return int(x) & 0xFFFFFF


def _u16(x: int) -> int:
    return int(x) & 0xFFFF


def _u8(x: int) -> int:
    return int(x) & 0xFF


# =============================================================================
# Function-face state update rules (first principles)
# =============================================================================


def step_state_l1(state8: int, byte_val: int) -> int:
    """L1 (8-bit) state update: XOR parity accumulation in intron space."""
    return _u8(state8) ^ _u8(intron(byte_val))


def step_state_l2(state16: int, byte_val: int) -> int:
    """
    L2 (16-bit) state update with chirality.

    state16 = (A8<<8) | B8

    x      = intron(byte)
    A_mut  = A8 XOR x
    A_next = B8 XOR 0xFF
    B_next = A_mut XOR 0xFF
    """
    s = _u16(state16)
    a8 = (s >> 8) & 0xFF
    b8 = s & 0xFF

    x = _u8(intron(byte_val))
    a_mut = (a8 ^ x) & 0xFF
    a_next = (b8 ^ 0xFF) & 0xFF
    b_next = (a_mut ^ 0xFF) & 0xFF

    return _u16((a_next << 8) | b_next)


def inverse_step_state_l2(state16: int, byte_val: int) -> int:
    """Exact inverse of L2 step."""
    s = _u16(state16)
    a_p = (s >> 8) & 0xFF
    b_p = s & 0xFF

    x = _u8(intron(byte_val))

    # predecessor:
    # B = A' XOR 0xFF
    # A = (B' XOR x) XOR 0xFF
    b = (a_p ^ 0xFF) & 0xFF
    a = ((b_p ^ x) ^ 0xFF) & 0xFF
    return _u16((a << 8) | b)


def step_state_l3(state24: int, byte_val: int) -> int:
    """
    L3 (24-bit) update: full GENE_Mac physics on (A12,B12), applied over full 2^24.

    m12   = mask12_for_byte(byte)
    A_mut = A12 XOR m12
    A'    = B12 XOR 0xFFF
    B'    = A_mut XOR 0xFFF
    """
    s = _u24(state24)
    a12, b12 = unpack_state(s)

    m12 = mask12_for_byte(_u8(byte_val)) & LAYER_MASK_12
    a_mut = (a12 ^ m12) & LAYER_MASK_12

    a_next = (b12 ^ LAYER_MASK_12) & LAYER_MASK_12
    b_next = (a_mut ^ LAYER_MASK_12) & LAYER_MASK_12

    return _u24(pack_state(a_next, b_next))


def inverse_step_state_l3(state24: int, byte_val: int) -> int:
    """
    Exact inverse of L3 step, valid over full 24-bit carrier.

    Given (A',B') and byte b with mask m:
      B = A' XOR 0xFFF
      A = (B' XOR m) XOR 0xFFF
    """
    s = _u24(state24)
    a_p, b_p = unpack_state(s)
    m12 = mask12_for_byte(_u8(byte_val)) & LAYER_MASK_12

    b = (a_p ^ LAYER_MASK_12) & LAYER_MASK_12
    a = ((b_p ^ m12) ^ LAYER_MASK_12) & LAYER_MASK_12
    return _u24(pack_state(a, b))


# =============================================================================
# Specs
# =============================================================================


@dataclass(frozen=True)
class FSMSpec:
    name: str
    state_bits: int
    num_states: int
    num_actions: int = 256

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_states, self.num_actions)

    @property
    def nparams(self) -> int:
        return self.num_states * self.num_actions

    def table_bytes(self, dtype: np.dtype) -> int:
        return int(self.nparams * np.dtype(dtype).itemsize)

    def table_gib(self, dtype: np.dtype) -> float:
        return self.table_bytes(dtype) / (1024.0**3)


L1_SPEC = FSMSpec("L1", 8, 1 << 8)
L2_SPEC = FSMSpec("L2", 16, 1 << 16)
L3_SPEC = FSMSpec("L3", 24, 1 << 24)

L2_ARCHETYPE_STATE16 = 0xAA55
L1_ARCHETYPE_STATE8 = 0x00
L3_ARCHETYPE_STATE24 = ARCHETYPE_STATE24  # 0xAAA555


# =============================================================================
# L1 FSM (uint8 table)
# =============================================================================


class Layer1FSM:
    """L1: 256x256 uint8 transition table."""

    spec = L1_SPEC
    archetype_state8 = L1_ARCHETYPE_STATE8

    def __init__(self, table: NDArray[np.uint8]):
        tab = np.asarray(table, dtype=np.uint8)
        if tab.shape != self.spec.shape:
            raise ValueError(f"L1 table shape {tab.shape}, expected {self.spec.shape}")
        self.table = tab

    @classmethod
    def build(cls) -> "Layer1FSM":
        # Vectorized build:
        # next[s,b] = s XOR intron(b), with intron(b) = b XOR 0xAA
        s = np.arange(256, dtype=np.uint16)[:, None]              # [256,1]
        intr = (np.arange(256, dtype=np.uint16) ^ GENE_MIC_S)[None, :]  # [1,256]
        tab = (s ^ intr).astype(np.uint8)                         # [256,256]
        return cls(tab)

    def next_state(self, state8: int, byte_val: int) -> int:
        return int(self.table[_u8(state8), _u8(byte_val)])

    def next_state_batch(self, states8: NDArray[np.uint8], byte_val: int) -> NDArray[np.uint8]:
        b = _u8(byte_val)
        return self.table[np.asarray(states8, dtype=np.uint8), b]

    def verify_matches_function(self) -> None:
        s = np.arange(256, dtype=np.uint16)[:, None]
        intr = (np.arange(256, dtype=np.uint16) ^ GENE_MIC_S)[None, :]
        expected = (s ^ intr).astype(np.uint8)
        if not np.array_equal(self.table, expected):
            raise AssertionError("L1 table does not match function face")

    def verify_columns_are_permutations(self) -> None:
        # For each byte b, mapping s->next is XOR with constant intron(b), hence a permutation.
        # Still, verify explicitly once.
        for b in range(256):
            col = self.table[:, b]
            if np.unique(col).size != 256:
                raise AssertionError(f"L1 byte {b} column is not a permutation")


# =============================================================================
# L2 FSM (uint16 table, vectorized build)
# =============================================================================


class Layer2FSM:
    """L2: 65536x256 uint16 transition table."""

    spec = L2_SPEC
    archetype_state16 = L2_ARCHETYPE_STATE16

    def __init__(self, table: NDArray[np.uint16]):
        tab = np.asarray(table, dtype=np.uint16)
        if tab.shape != self.spec.shape:
            raise ValueError(f"L2 table shape {tab.shape}, expected {self.spec.shape}")
        self.table = tab

    @classmethod
    def build(cls) -> "Layer2FSM":
        """
        Vectorized build.

        Uses:
          states: 0..65535
          a8 = states>>8, b8 = states&0xFF
          for each byte b:
            x = intron(b)
            a_mut = a8 ^ x
            a_next = b8 ^ 0xFF
            b_next = a_mut ^ 0xFF
            next = (a_next<<8) | b_next
        """
        states = np.arange(1 << 16, dtype=np.uint32)
        a8 = ((states >> 8) & 0xFF).astype(np.uint8)
        b8 = (states & 0xFF).astype(np.uint8)

        intr_by_byte = (np.arange(256, dtype=np.uint16) ^ GENE_MIC_S).astype(np.uint8)

        tab = np.empty((1 << 16, 256), dtype=np.uint16)
        a_next = (b8 ^ 0xFF).astype(np.uint8)  # independent of byte

        for b in range(256):
            x = intr_by_byte[b]
            a_mut = (a8 ^ x).astype(np.uint8)
            b_next = (a_mut ^ 0xFF).astype(np.uint8)
            tab[:, b] = ((a_next.astype(np.uint16) << 8) | b_next.astype(np.uint16))

        return cls(tab)

    def next_state(self, state16: int, byte_val: int) -> int:
        return int(self.table[_u16(state16), _u8(byte_val)])

    def next_state_batch(self, states16: NDArray[np.uint16], byte_val: int) -> NDArray[np.uint16]:
        b = _u8(byte_val)
        return self.table[np.asarray(states16, dtype=np.uint16), b]

    def verify_matches_function(self, samples: int = 200_000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        s = rng.integers(0, 1 << 16, size=samples, dtype=np.uint32)
        b = rng.integers(0, 256, size=samples, dtype=np.uint32)
        tab_next = self.table[s.astype(np.uint16), b.astype(np.uint8)].astype(np.uint16)

        fn_next = np.empty(samples, dtype=np.uint16)
        for i in range(samples):
            fn_next[i] = step_state_l2(int(s[i]), int(b[i]))

        if not np.array_equal(tab_next, fn_next):
            raise AssertionError("L2 table does not match function face on samples")

    def verify_inverse(self, samples: int = 200_000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        s = rng.integers(0, 1 << 16, size=samples, dtype=np.uint32)
        b = rng.integers(0, 256, size=samples, dtype=np.uint32)
        s_next = self.table[s.astype(np.uint16), b.astype(np.uint8)].astype(np.uint16)

        s_pred = np.empty(samples, dtype=np.uint16)
        for i in range(samples):
            s_pred[i] = inverse_step_state_l2(int(s_next[i]), int(b[i]))

        if not np.array_equal(s.astype(np.uint16), s_pred):
            raise AssertionError("L2 inverse_step failed on samples")


# =============================================================================
# L3 FSM (packed uint24 table, memmap recommended)
# =============================================================================


class Layer3FSM:
    """
    L3 packed table:
      file-backed uint8 array of shape (2^24, 256, 3)
    where the 3 bytes encode next_state24 in little-endian:
      lo = x & 0xFF
      mi = (x >> 8) & 0xFF
      hi = (x >> 16) & 0xFF
    """

    spec = L3_SPEC
    archetype_state24 = L3_ARCHETYPE_STATE24

    def __init__(self, table_u8: NDArray[np.uint8]):
        tab = np.asarray(table_u8, dtype=np.uint8)
        if tab.shape != (self.spec.num_states, 256, 3):
            raise ValueError(
                f"L3 packed table shape {tab.shape}, expected {(self.spec.num_states, 256, 3)}"
            )
        self.table = tab

        # precompute masks (fast build and fast function face)
        self.mask12_by_byte = np.array([mask12_for_byte(b) & 0xFFF for b in range(256)], dtype=np.uint16)

    @classmethod
    def from_memmap(
        cls, path: Path, mode: Literal["r", "r+", "c"] = "r"
    ) -> "Layer3FSM":
        path = Path(path)
        tab: NDArray[np.uint8] = np.memmap(
            str(path),
            dtype=np.uint8,
            mode=mode,
            shape=(1 << 24, 256, 3),
        )  # type: ignore[call-overload]
        return cls(tab)

    @staticmethod
    def _pack_u24_le(x: NDArray[np.uint32]) -> NDArray[np.uint8]:
        """Pack uint32 array (values < 2^24) into bytes [*,3] little-endian."""
        out = np.empty((x.size, 3), dtype=np.uint8)
        out[:, 0] = (x & 0xFF).astype(np.uint8)
        out[:, 1] = ((x >> 8) & 0xFF).astype(np.uint8)
        out[:, 2] = ((x >> 16) & 0xFF).astype(np.uint8)
        return out

    @staticmethod
    def _unpack_u24_le(b3: NDArray[np.uint8]) -> NDArray[np.uint32]:
        """Unpack bytes [*,3] little-endian into uint32."""
        b3 = np.asarray(b3, dtype=np.uint8)
        raw = (
            b3[:, 0].astype(np.uint32)
            | (b3[:, 1].astype(np.uint32) << 8)
            | (b3[:, 2].astype(np.uint32) << 16)
        )
        return raw.astype(np.uint32)

    @classmethod
    def build_memmap_packed(
        cls,
        path: Path,
        chunk_states: int = 1 << 16,
        progress_every_chunks: int = 64,
    ) -> "Layer3FSM":
        """
        Build packed L3 transition table as uint8 memmap [2^24,256,3].

        This writes 12.9 GiB of transition data. It is finite but heavy.
        Use only if you truly want table-face speed for L3.

        The build is chunked over state space; within each chunk we vectorize over states.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tab = np.memmap(
            str(path),
            dtype=np.uint8,
            mode="w+",
            shape=(1 << 24, 256, 3),
        )

        # precompute masks for 256 bytes
        mask12_by_byte = np.array([mask12_for_byte(b) & 0xFFF for b in range(256)], dtype=np.uint16)

        n_states = 1 << 24
        t0 = time.time()

        for chunk_idx, start in enumerate(range(0, n_states, chunk_states)):
            end = min(start + chunk_states, n_states)
            states = np.arange(start, end, dtype=np.uint32)

            a = (states >> 12) & 0xFFF
            b = states & 0xFFF

            # a_next is independent of byte: A_next = B ^ 0xFFF
            a_next = (b ^ 0xFFF).astype(np.uint32)

            for byte_val in range(256):
                m = np.uint32(mask12_by_byte[byte_val])
                a_mut = (a ^ m) & 0xFFF
                b_next = (a_mut ^ 0xFFF).astype(np.uint32)
                ns = ((a_next << 12) | b_next).astype(np.uint32)

                packed = cls._pack_u24_le(ns)  # [block,3]
                tab[start:end, byte_val, :] = packed

            if (chunk_idx % progress_every_chunks) == 0:
                elapsed = time.time() - t0
                pct = 100.0 * end / n_states
                print(f"L3 build: {end:,}/{n_states:,} ({pct:.1f}%) elapsed={elapsed:.1f}s")
                tab.flush()

        tab.flush()
        size_gib = path.stat().st_size / (1024.0**3)
        print(f"L3 packed table built at {path} ({size_gib:.2f} GiB)")
        return cls(tab)

    # ----- table face -----

    def next_state(self, state24: int, byte_val: int) -> int:
        b3 = self.table[_u24(state24), _u8(byte_val), :]  # [3]
        return int(b3[0]) | (int(b3[1]) << 8) | (int(b3[2]) << 16)

    def fanout(self, state24: int) -> NDArray[np.uint32]:
        b3 = self.table[_u24(state24), :, :]  # [256,3]
        return self._unpack_u24_le(b3)

    # ----- function face -----

    def next_state_functional(self, state24: int, byte_val: int) -> int:
        return step_state_l3(state24, byte_val)

    def inverse_state_functional(self, state24: int, byte_val: int) -> int:
        return inverse_step_state_l3(state24, byte_val)

    def verify_inverse_samples(self, samples: int = 200_000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        s = rng.integers(0, 1 << 24, size=samples, dtype=np.uint32)
        b = rng.integers(0, 256, size=samples, dtype=np.uint32)
        for i in range(samples):
            s0 = int(s[i])
            bb = int(b[i])
            s1 = self.next_state(s0, bb)
            sp = inverse_step_state_l3(s1, bb)
            if sp != (s0 & 0xFFFFFF):
                raise AssertionError("L3 inverse check failed")

    def verify_table_matches_function_samples(self, samples: int = 200_000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        s = rng.integers(0, 1 << 24, size=samples, dtype=np.uint32)
        b = rng.integers(0, 256, size=samples, dtype=np.uint32)
        for i in range(samples):
            s0 = int(s[i])
            bb = int(b[i])
            t_next = self.next_state(s0, bb)
            f_next = step_state_l3(s0, bb)
            if t_next != f_next:
                raise AssertionError("L3 table != function on samples")


# =============================================================================
# L4 dual operator (small holographic register + helpers)
# =============================================================================


@dataclass
class L4State:
    """
    Tiny holographic register over the byte stream (12-bit mask space).

    O: XOR of mask12 at positions 0,2,4,... (even index / "odd slot" in prior naming)
    E: XOR of mask12 at positions 1,3,5,...
    parity: length % 2
    length: number of bytes processed
    """
    O: int = 0
    E: int = 0
    parity: int = 0
    length: int = 0

    def reset(self) -> None:
        self.O = 0
        self.E = 0
        self.parity = 0
        self.length = 0


class Layer4:
    """
    Dual operator: stream closure + reconstruction utilities.
    Not a table.

    Provided:
      - update(state, byte): update O/E commitments
      - commitment(seq): compute (O,E,parity) for a sequence
      - equivalent(seq_a, seq_b): P8 equivalence check
      - identity_cycle(state): check O==0 and E==0 and parity==0
      - closure_score(state): graded closure 0..1 from popcount(O)+popcount(E)
      - inverse_l3(state24, byte): exact inverse stepping (function face)
    """

    def __init__(self) -> None:
        self.mask12_by_byte = np.array([mask12_for_byte(b) & 0xFFF for b in range(256)], dtype=np.uint16)

    def update(self, st: L4State, byte_val: int) -> None:
        b = _u8(byte_val)
        m = int(self.mask12_by_byte[b])
        if (st.length & 1) == 0:
            st.O ^= m
        else:
            st.E ^= m
        st.length += 1
        st.parity = st.length & 1

    def commitment(self, seq: bytes | list[int]) -> Tuple[int, int, int]:
        O = 0
        E = 0
        for i, bb in enumerate(seq):
            m = int(self.mask12_by_byte[_u8(bb)])
            if (i & 1) == 0:
                O ^= m
            else:
                E ^= m
        return O, E, (len(seq) & 1)

    def equivalent(self, seq_a: bytes | list[int], seq_b: bytes | list[int]) -> bool:
        return self.commitment(seq_a) == self.commitment(seq_b)

    def identity_cycle(self, st: L4State) -> bool:
        return (st.O == 0) and (st.E == 0) and (st.parity == 0)

    def closure_score(self, st: L4State) -> float:
        # 24 total bits of (O,E). Normalize to [0,1].
        w = popcount(st.O & 0xFFF) + popcount(st.E & 0xFFF)
        return float(w) / 24.0

    def inverse_l3(self, state24: int, byte_val: int) -> int:
        return inverse_step_state_l3(state24, byte_val)


# =============================================================================
# Bundling for agents
# =============================================================================


@dataclass
class LayerRegisters:
    """Per-agent registers for L1/L2/L3 + L4 holographic register."""
    l1_state8: int = L1_ARCHETYPE_STATE8
    l2_state16: int = L2_ARCHETYPE_STATE16
    l3_state24: int = L3_ARCHETYPE_STATE24
    l4: L4State = field(default_factory=L4State)

    def reset(self) -> None:
        self.l1_state8 = L1_ARCHETYPE_STATE8
        self.l2_state16 = L2_ARCHETYPE_STATE16
        self.l3_state24 = L3_ARCHETYPE_STATE24
        self.l4.reset()


@dataclass
class FourLayers:
    """Concrete bundle of L1/L2/L3 FSMs + L4 dual operator, plus per-agent registers."""
    l1: Layer1FSM
    l2: Layer2FSM
    l3: Layer3FSM
    l4: Layer4
    regs: LayerRegisters

    def reset(self) -> None:
        self.regs.reset()

    def ingest_byte(self, byte_val: int) -> None:
        b = _u8(byte_val)
        self.regs.l1_state8 = self.l1.next_state(self.regs.l1_state8, b)
        self.regs.l2_state16 = self.l2.next_state(self.regs.l2_state16, b)
        self.regs.l3_state24 = self.l3.next_state(self.regs.l3_state24, b)
        self.l4.update(self.regs.l4, b)


def create_default_four_layers(
    *,
    l3_path: Optional[Path] = None,
    build_l3_if_missing: bool = False,
) -> FourLayers:
    """
    Convenience factory:

      - builds L1 table in RAM (64 KB)
      - builds L2 table in RAM (32 MB, vectorized)
      - loads L3 packed memmap from l3_path, or builds it if requested
      - creates L4 operator and per-agent registers

    L3 packed table file size: ~12.9 GiB.
    """
    l1 = Layer1FSM.build()
    l1.verify_matches_function()

    l2 = Layer2FSM.build()
    # Optional: spot-check correctness (fast enough to run once)
    l2.verify_inverse(samples=50_000)

    if l3_path is None:
        raise ValueError("l3_path is required (packed L3 table lives on disk)")

    l3_path = Path(l3_path)
    if l3_path.exists():
        l3 = Layer3FSM.from_memmap(l3_path, mode="r")
    else:
        if not build_l3_if_missing:
            raise FileNotFoundError(f"L3 packed table not found at {l3_path}")
        l3 = Layer3FSM.build_memmap_packed(l3_path)

    l4 = Layer4()
    regs = LayerRegisters()
    return FourLayers(l1=l1, l2=l2, l3=l3, l4=l4, regs=regs)


__all__ = [
    "FSMSpec",
    "Layer1FSM",
    "Layer2FSM",
    "Layer3FSM",
    "Layer4",
    "L4State",
    "LayerRegisters",
    "FourLayers",
    "L1_SPEC",
    "L2_SPEC",
    "L3_SPEC",
    "step_state_l1",
    "step_state_l2",
    "step_state_l3",
    "inverse_step_state_l2",
    "inverse_step_state_l3",
    "create_default_four_layers",
]