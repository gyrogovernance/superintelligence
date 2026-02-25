"""
src.tools.layers
================

Three NEW finite state machine layers plus one holographic closure
layer, independent of the legacy Router atlas.

They do NOT touch:
    - ontology.npy
    - epistemology.npy
    - phenomenology.npz
or RouterKernel at runtime.

Each of L1, L2, L3 is a true state machine:

    Layer 1 (L1):  8-bit state
        256 states  × 256 input bytes  = 65,536 transitions

    Layer 2 (L2): 16-bit state
        65,536 states × 256 input bytes = 16,777,216 transitions

    Layer 3 (L3): 24-bit state
        16,777,216 states × 256 input bytes = 4,294,967,296 transitions

Each transition table stores the next state as an integer index.
State-update functions are derived from GENE_Mac physics but scaled
to the appropriate bit-width.

Layer 4 (BU) is NOT a table. It is a small holographic closure
operator over byte sequences and 24-bit states (trajectory parity,
inverse stepping, etc).

These are the “4 layers” that agents can use as structural memory.

Design:
    - L1/L2 fully in RAM by default.
    - L3 is large; usually created as a memmap (on disk), but it is
      still a real table with a real byte size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_B12,
    ARCHETYPE_STATE24,
    GENE_MIC_S,
    LAYER_MASK_12,
    mask12_for_byte,
    pack_state,
    popcount,
    unpack_state,
)

# ---------------------------------------------------------------------
# Shared physics primitives
# ---------------------------------------------------------------------


def intron(byte_val: int) -> int:
    """Transcription intron = byte XOR 0xAA."""
    return (int(byte_val) & 0xFF) ^ GENE_MIC_S


# --------------- L1: 8-bit state update ------------------------------


def step_state_l1(state8: int, byte_val: int) -> int:
    """
    L1 state update: XOR parity in intron space.

    This is the minimal CS-level “memory” of the byte history.
    """
    return (int(state8) ^ intron(byte_val)) & 0xFF


# --------------- L2: 16-bit state update -----------------------------


def step_state_l2(state16: int, byte_val: int) -> int:
    """
    L2 state update: 16-bit chirality analogue of GENE_Mac.

    State is (A8,B8), packed into 16 bits:

        state16 = (A8 << 8) | B8

    Update rule (same pattern as 24-bit GENE_Mac, but on 8 bits):
        x      = intron(byte)
        A_mut  = A8 XOR x
        A_next = B8 XOR 0xFF
        B_next = A_mut XOR 0xFF
    """
    s = int(state16) & 0xFFFF
    a8 = (s >> 8) & 0xFF
    b8 = s & 0xFF

    x = intron(byte_val) & 0xFF
    a_mut = (a8 ^ x) & 0xFF
    a_next = (b8 ^ 0xFF) & 0xFF
    b_next = (a_mut ^ 0xFF) & 0xFF

    return ((a_next << 8) | b_next) & 0xFFFF


# --------------- L3: 24-bit state update -----------------------------


def step_state_l3(state24: int, byte_val: int) -> int:
    """
    L3 state update: full GENE_Mac physics on (A12, B12).

        intron = byte XOR 0xAA
        mask12 = expand(intron)   # via mask12_for_byte
        A_mut  = A12 XOR mask12
        A'     = B12 XOR 0xFFF
        B'     = A_mut XOR 0xFFF

    Here we reuse mask12_for_byte, which already encodes the canonical
    intron → mask expansion.
    """
    s = int(state24) & 0xFFFFFF
    a12, b12 = unpack_state(s)

    m12 = mask12_for_byte(byte_val & 0xFF) & LAYER_MASK_12
    a_mut = (a12 ^ m12) & LAYER_MASK_12

    a_next = (b12 ^ LAYER_MASK_12) & LAYER_MASK_12
    b_next = (a_mut ^ LAYER_MASK_12) & LAYER_MASK_12

    return pack_state(a_next, b_next) & 0xFFFFFF


def inverse_step_state_l3(state24: int, byte_val: int) -> int:
    """
    Exact inverse of L3 step (P9 form):

        B = A' XOR 0xFFF
        A = (B' XOR mask12) XOR 0xFFF
    """
    s = int(state24) & 0xFFFFFF
    a_p, b_p = unpack_state(s)
    m12 = mask12_for_byte(byte_val & 0xFF) & LAYER_MASK_12

    B = (a_p ^ LAYER_MASK_12) & LAYER_MASK_12
    A = ((b_p ^ m12) ^ LAYER_MASK_12) & LAYER_MASK_12
    return pack_state(A, B) & 0xFFFFFF


# ---------------------------------------------------------------------
# Table specs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FSMSpec:
    name: str
    state_bits: int
    num_states: int
    num_actions: int = 256

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_states, self.num_actions)

    def nparams(self) -> int:
        return self.num_states * self.num_actions

    def nbytes(self, dtype: np.dtype) -> int:
        return int(self.nparams() * np.dtype(dtype).itemsize)

    def nbytes_gb(self, dtype: np.dtype) -> float:
        return self.nbytes(dtype) / (1024.0**3)


L1_SPEC = FSMSpec(name="L1", state_bits=8, num_states=1 << 8)
L2_SPEC = FSMSpec(name="L2", state_bits=16, num_states=1 << 16)
L3_SPEC = FSMSpec(name="L3", state_bits=24, num_states=1 << 24)


# ---------------------------------------------------------------------
# FSM layers: each holds a transition table next_state[state, byte]
# ---------------------------------------------------------------------


class Layer1FSM:
    """
    Layer 1 (L1): 8-bit FSM

    - 256 states
    - Transition table: 256 × 256
    - Each entry is a uint8 next-state.

    State semantics: cumulative XOR parity of introns so far.
    Archetype state: 0x00 (no mutations).
    """

    spec = L1_SPEC

    def __init__(self, table: NDArray[np.uint8]):
        table = np.asarray(table, dtype=np.uint8)
        if table.shape != self.spec.shape:
            raise ValueError(f"L1 table shape {table.shape}, expected {self.spec.shape}")
        self.table: NDArray[np.uint8] = table

    @classmethod
    def build(cls) -> "Layer1FSM":
        """
        Build the full 256×256 table via step_state_l1.
        This is cheap and can be done eagerly in RAM.
        """
        tab = np.empty(cls.spec.shape, dtype=np.uint8)
        for s in range(cls.spec.num_states):
            for b in range(256):
                tab[s, b] = step_state_l1(s, b)
        return cls(tab)

    def next_state(self, state: int, byte_val: int) -> int:
        return int(self.table[state & 0xFF, byte_val & 0xFF])

    def run(self, byte_sequence: bytes | list[int], start_state: int = 0) -> list[int]:
        """Return the L1 state trajectory for a byte sequence."""
        s = start_state & 0xFF
        traj = [s]
        for b in byte_sequence:
            s = self.next_state(s, b)
            traj.append(s)
        return traj

    @property
    def nparams(self) -> int:
        return self.spec.nparams()

    @property
    def nbytes(self) -> int:
        return int(self.table.nbytes)


class Layer2FSM:
    """
    Layer 2 (L2): 16-bit FSM

    - 65,536 states
    - Transition table: 65,536 × 256
    - Each entry is a uint16 next-state.

    State semantics: (A8,B8) with chirality:
      A_next = B8 XOR 0xFF
      B_next = (A8 XOR intron) XOR 0xFF

    Archetype: A8=0xAA, B8=0x55 → 0xAA55.
    """

    spec = L2_SPEC
    ARCHETYPE_STATE16: int = 0xAA55

    def __init__(self, table: NDArray[np.uint16]):
        table = np.asarray(table, dtype=np.uint16)
        if table.shape != self.spec.shape:
            raise ValueError(f"L2 table shape {table.shape}, expected {self.spec.shape}")
        self.table: NDArray[np.uint16] = table

    @classmethod
    def build(cls, tqdm: bool = False) -> "Layer2FSM":
        """
        Build the full 65,536×256 table via step_state_l2.

        Naive double loop; for production, vectorisation over chunks
        is recommended, but this is correct and finite.
        """
        tab = np.empty(cls.spec.shape, dtype=np.uint16)
        rng = range(cls.spec.num_states)
        if tqdm:
            try:
                from tqdm import tqdm as _tqdm
                rng = _tqdm(rng)
            except Exception:
                pass

        for s in rng:
            for b in range(256):
                tab[s, b] = step_state_l2(s, b)
        return cls(tab)

    def next_state(self, state: int, byte_val: int) -> int:
        return int(self.table[state & 0xFFFF, byte_val & 0xFF])

    def run(self, byte_sequence: bytes | list[int], start_state: Optional[int] = None) -> list[int]:
        s = self.ARCHETYPE_STATE16 if start_state is None else (start_state & 0xFFFF)
        traj = [s]
        for b in byte_sequence:
            s = self.next_state(s, b)
            traj.append(s)
        return traj

    @property
    def nparams(self) -> int:
        return self.spec.nparams()

    @property
    def nbytes(self) -> int:
        return int(self.table.nbytes)


class Layer3FSM:
    """
    Layer 3 (L3): 24-bit FSM

    - 16,777,216 states
    - Transition table: 16,777,216 × 256
    - Each entry is a uint32 next-state (0..2^24-1).

    This is large: 16,777,216 × 256 × 4 bytes ≈ 16.8 GB.

    Typically, you will memory-map it to disk (np.memmap) so that
    it has real storage but does not need to be fully in RAM.
    """

    spec = L3_SPEC
    ARCHETYPE_STATE24: int = ARCHETYPE_STATE24  # 0xAAA555

    def __init__(self, table: NDArray[np.uint32]):
        table = np.asarray(table, dtype=np.uint32)
        if table.shape != self.spec.shape:
            raise ValueError(f"L3 table shape {table.shape}, expected {self.spec.shape}")
        self.table: NDArray[np.uint32] = table

    @classmethod
    def build_memmap(cls, path: Path) -> "Layer3FSM":
        """
        Build the full 24-bit transition table as a memmap on disk.

        WARNING: ~16.8 GB file. This is finite, but slow to build.

        Strategy:
          - Use open_memmap with shape (2^24, 256)
          - Chunk the state space (e.g., 2^16 per chunk)
          - Vectorise where possible
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        from numpy.lib.format import open_memmap

        n_states = cls.spec.num_states
        tab = open_memmap(
            str(path),
            mode="w+",
            dtype=np.uint32,
            shape=cls.spec.shape,
        )

        chunk = 1 << 16  # 65,536 states at a time
        t0 = time.time()

        for start in range(0, n_states, chunk):
            end = min(start + chunk, n_states)
            state_block = np.arange(start, end, dtype=np.uint32)

            # Decompose into A12,B12 for this block
            a_block = (state_block >> 12) & 0xFFF
            b_block = state_block & 0xFFF

            for b in range(256):
                # Step all states in this block with byte b
                m12 = mask12_for_byte(b) & LAYER_MASK_12
                a_mut = (a_block ^ m12) & LAYER_MASK_12
                a_next = (b_block ^ LAYER_MASK_12) & LAYER_MASK_12
                b_next = (a_mut ^ LAYER_MASK_12) & LAYER_MASK_12
                tab[start:end, b] = ((a_next << 12) | b_next).astype(np.uint32)

            if (end - start) >= chunk and (start // chunk) % 64 == 0:
                elapsed = time.time() - t0
                pct = 100.0 * end / n_states
                print(f"L3 build: {end:,}/{n_states:,} ({pct:.1f}%) in {elapsed:.1f}s")

            tab.flush()

        size_gb = path.stat().st_size / (1024.0**3)
        print(f"L3 table built at {path} ({size_gb:.2f} GB)")
        return cls(tab)

    @classmethod
    def from_memmap(cls, path: Path) -> "Layer3FSM":
        path = Path(path)
        arr = np.memmap(str(path), dtype=np.uint32, mode="r", shape=cls.spec.shape)
        return cls(arr)

    def next_state(self, state24: int, byte_val: int) -> int:
        return int(self.table[state24 & 0xFFFFFF, byte_val & 0xFF])

    def run(self, byte_sequence: bytes | list[int], start_state: Optional[int] = None) -> list[int]:
        s = self.ARCHETYPE_STATE24 if start_state is None else (start_state & 0xFFFFFF)
        traj = [s]
        for b in byte_sequence:
            s = self.next_state(s, b)
            traj.append(s)
        return traj

    @property
    def nparams(self) -> int:
        return self.spec.nparams()

    @property
    def nbytes(self) -> int:
        return int(self.table.nbytes)


# ---------------------------------------------------------------------
# Layer 4 — BU: holographic closure (no table)
# ---------------------------------------------------------------------


@dataclass
class BUState:
    """
    Small holographic compression of a byte path in mask-space.

    Stores:
        O = XOR of 12-bit masks at odd positions
        E = XOR of 12-bit masks at even positions
        parity = length mod 2
        length = number of bytes seen
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


class BULayer:
    """
    Balance Universal (BU) holography layer.

    - No parameter table.
    - Provides:
        * egress_update: update (O,E,parity,length) given a new byte
        * commitment: compute (O,E,parity) for a whole sequence
        * equivalent: P8-style path equivalence
        * inverse_step_state_l3: exact inverse at 24-bit
    """

    def __init__(self) -> None:
        self.mask12_by_byte = np.array(
            [mask12_for_byte(b) & LAYER_MASK_12 for b in range(256)],
            dtype=np.uint16,
        )

    def egress_update(self, bu: BUState, byte_val: int) -> None:
        """Update (O,E,parity,length) with a new byte."""
        b = int(byte_val) & 0xFF
        m = int(self.mask12_by_byte[b])

        if bu.length % 2 == 0:
            bu.O ^= m
        else:
            bu.E ^= m

        bu.length += 1
        bu.parity = bu.length & 1

    def commitment(self, byte_sequence: bytes | list[int]) -> tuple[int, int, int]:
        """Return (O,E,parity) without mutating a BUState."""
        O = 0
        E = 0
        for i, b in enumerate(byte_sequence):
            m = int(self.mask12_by_byte[int(b) & 0xFF])
            if i % 2 == 0:
                O ^= m
            else:
                E ^= m
        return O, E, (len(byte_sequence) & 1)

    def equivalent(self, seq_a: bytes | list[int], seq_b: bytes | list[int]) -> bool:
        """P8 path equivalence: same (O,E,parity) => same effect from any start."""
        return self.commitment(seq_a) == self.commitment(seq_b)

    def inverse_step_state24(self, state24: int, byte_val: int) -> int:
        """Exact inverse of L3 step (full 24-bit)."""
        return inverse_step_state_l3(state24, byte_val)


# ---------------------------------------------------------------------
# Bundling for agents
# ---------------------------------------------------------------------


@dataclass
class LayerRegisters:
    """Per-agent registers for the 3 FSM layers + BU commitments."""
    l1_state8: int = 0x00
    l2_state16: int = Layer2FSM.ARCHETYPE_STATE16
    l3_state24: int = Layer3FSM.ARCHETYPE_STATE24
    bu: BUState = field(default_factory=BUState)

    def reset(self) -> None:
        self.l1_state8 = 0x00
        self.l2_state16 = Layer2FSM.ARCHETYPE_STATE16
        self.l3_state24 = Layer3FSM.ARCHETYPE_STATE24
        self.bu.reset()


@dataclass
class FourFSMs:
    """
    Concrete bundle of the 3 FSM layers + BU holography, plus registers.

    This is what an agent can own as its structural memory substrate.
    """
    l1: Layer1FSM
    l2: Layer2FSM
    l3: Layer3FSM
    bu: BULayer
    regs: LayerRegisters

    def reset(self) -> None:
        self.regs.reset()

    def ingest_byte(self, byte_val: int) -> None:
        """Update all three FSM states + BU commitments."""
        b = int(byte_val) & 0xFF
        self.regs.l1_state8 = self.l1.next_state(self.regs.l1_state8, b)
        self.regs.l2_state16 = self.l2.next_state(self.regs.l2_state16, b)
        self.regs.l3_state24 = self.l3.next_state(self.regs.l3_state24, b)
        self.regs.bu.egress_update(self.regs.bu, b)


def create_default_four_fsms(
    *,
    l3_path: Optional[Path] = None,
    build_l3_if_missing: bool = False,
) -> FourFSMs:
    """
    Convenience factory:

        - Builds L1 and L2 tables in RAM.
        - Loads L3 from memmap at l3_path, or builds it if requested.
        - Creates BU operator and zeroed registers.

    L3 is the heavy one (~16.8 GB). If you do not want to
    materialise it yet, you can skip calling this or set
    build_l3_if_missing=False and ensure l3_path exists.
    """
    print("Building L1 FSM (256×256)...")
    l1 = Layer1FSM.build()
    print(f"  L1 params: {l1.nparams:,}, bytes: {l1.nbytes:,}")

    print("Building L2 FSM (65,536×256)...")
    t0 = time.time()
    l2 = Layer2FSM.build(tqdm=True)
    print(
        f"  L2 params: {l2.nparams:,}, bytes: {l2.nbytes / (1024**2):.1f} MB,"
        f" built in {time.time() - t0:.1f}s"
    )

    if l3_path is None:
        raise ValueError("l3_path is required to create/load L3 FSM")

    l3_path = Path(l3_path)

    if l3_path.exists():
        print(f"Loading L3 FSM from {l3_path}...")
        l3 = Layer3FSM.from_memmap(l3_path)
    else:
        if not build_l3_if_missing:
            raise FileNotFoundError(
                f"L3 FSM file not found at {l3_path} and build_l3_if_missing=False"
            )
        print("Building L3 FSM (~16.8 GB) — this will take a while...")
        l3 = Layer3FSM.build_memmap(l3_path)

    bu = BULayer()
    regs = LayerRegisters()
    return FourFSMs(l1=l1, l2=l2, l3=l3, bu=bu, regs=regs)


__all__ = [
    "FSMSpec",
    "Layer1FSM",
    "Layer2FSM",
    "Layer3FSM",
    "BULayer",
    "BUState",
    "LayerRegisters",
    "FourFSMs",
    "L1_SPEC",
    "L2_SPEC",
    "L3_SPEC",
    "step_state_l1",
    "step_state_l2",
    "step_state_l3",
    "inverse_step_state_l3",
    "create_default_four_fsms",
]