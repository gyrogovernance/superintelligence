"""
Deterministic governance operations for the GGG ASI Alignment Router.

This module defines the byte-to-action transcription law and the 48-bit tensor
representation of governance states. The algebra is deterministic and serves as the
single source of truth for Router state representation and transitions.
"""


import numpy as np
from numpy.typing import NDArray


GENE_Mic_S = 0xAA  # 10101010

# Archetypal 48-bit tensor [4, 2, 3, 2] with alternating ±1
GENE_Mac_S = np.array(
    [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],  # Layer 0
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],  # Layer 1
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],  # Layer 2
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],  # Layer 3
    ],
    dtype=np.int8,
)

# Exon families (bit classes)
EXON_LI_MASK = 0b01000010  # UNA   bits (Parity / Reflection)
EXON_FG_MASK = 0b00100100  # ONA   bits (Forward Gyration)
EXON_BG_MASK = 0b00011000  # BU-Eg bits (Backward Gyration)


def _build_masks_and_constants() -> tuple[int, int, int, list[int]]:
    """
    Compute FG/BG masks (layer-select) and action broadcast patterns.

    Returns a tuple (FG_MASK, BG_MASK, FULL_MASK, ACTION_BROADCAST_MASKS_LIST).
    """
    FG, BG = 0, 0
    # Flatten order = C; bit index: ((layer*2 + frame)*3 + row)*2 + col
    for layer in range(4):
        for frame in range(2):
            for row in range(3):
                for col in range(2):
                    bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col
                    if layer in (0, 2):
                        FG |= 1 << bit_index
                    if layer in (1, 3):
                        BG |= 1 << bit_index
    FULL_MASK = (1 << 48) - 1

    action_broadcast_masks_list: list[int] = []
    for i in range(256):
        # broadcast byte across 6 bytes (to 48 bits)
        mask = 0
        for j in range(6):
            mask |= i << (8 * j)
        action_broadcast_masks_list.append(mask)

    return FG, BG, FULL_MASK, action_broadcast_masks_list


FG_MASK, BG_MASK, FULL_MASK, ACTION_BROADCAST_MASKS_LIST = _build_masks_and_constants()
ACTION_BROADCAST_MASKS: NDArray[np.uint64] = np.array(ACTION_BROADCAST_MASKS_LIST, dtype=np.uint64)

# Layer masks: each layer has 12 bits (2 frames × 3 rows × 2 cols)
# Tensor is flattened in C-order: [layer, frame, row, col] -> position in flattened array
# When packed to integer: bits[position] -> integer bit (47 - position)
# So for integer bit_index: position_in_array = 47 - bit_index
# For shape [4, 2, 3, 2], C-order: position = layer*12 + frame*6 + row*2 + col
LAYER_MASKS = np.zeros(4, dtype=np.uint64)
for layer in range(4):
    mask = 0
    for frame in range(2):
        for row in range(3):
            for col in range(2):
                # C-order flattening: layer is the slowest dimension
                # Strides: layer=12, frame=6, row=2, col=1
                position_in_array = layer * 12 + frame * 6 + row * 2 + col
                # Integer bit index: MSB (bit 47) corresponds to position 0
                integer_bit_index = 47 - position_in_array
                if integer_bit_index < 0 or integer_bit_index >= 48:
                    continue  # Safety check
                mask |= 1 << integer_bit_index
    LAYER_MASKS[layer] = np.uint64(mask)

# BU loop internal actions (direct indices into epistemology, not external bytes)
# These are the internal action values for the three commutator loops
UNA_P_ACTION = 0x02  # UNA positive (internal action)
ONA_P_ACTION = 0x04  # ONA positive (internal action)
BU_P_ACTION = 0x08   # BU positive (internal action)
BU_M_ACTION = 0x10   # BU negative (internal action)
ONA_M_ACTION = 0x20  # ONA negative (internal action)
UNA_M_ACTION = 0x40  # UNA negative (internal action)

# Transform mask per action (precompute once)
XFORM_MASK = np.empty(256, dtype=np.uint64)
for i in range(256):
    m = 0
    if i & EXON_LI_MASK:
        m ^= FULL_MASK
    if i & EXON_FG_MASK:
        m ^= FG_MASK
    if i & EXON_BG_MASK:
        m ^= BG_MASK
    XFORM_MASK[i] = m


def tensor_to_int(tensor: NDArray[np.int8]) -> int:
    """Map ±1 tensor [4,2,3,2] to a 48-bit big-endian integer."""
    if tensor.shape != (4, 2, 3, 2):
        raise ValueError(f"Expected tensor shape (4,2,3,2), got {tensor.shape}")
    bits = (tensor.flatten(order="C") == -1).astype(np.uint8)
    packed = np.packbits(bits, bitorder="big")
    return int.from_bytes(packed.tobytes(), "big")


def int_to_tensor(state: int) -> NDArray[np.int8]:
    """Convert a 48-bit integer back to a tensor of shape [4,2,3,2]."""
    bits = [(state >> i) & 1 for i in range(47, -1, -1)]
    tensor_flat = np.array([1 if bit == 0 else -1 for bit in bits], dtype=np.int8)
    return tensor_flat.reshape((4, 2, 3, 2), order="C")


def byte_to_action(byte: int) -> int:
    """Deterministic byte to action transcription via XOR with GENE_Mic_S."""
    return (byte & 0xFF) ^ GENE_Mic_S




def apply_transition(state_int: int, action_byte: int) -> int:
    """
    Apply a single-step state transform in the 48-bit manifold.

    This provides the reference physics for generating atlas transitions but
    is not used directly by the runtime Router kernel, which relies on the
    precomputed epistemology table.
    """
    state_int = int(state_int) & ((1 << 48) - 1)
    ii = int(action_byte) & 0xFF
    temp = state_int ^ int(XFORM_MASK[ii])
    pattern = int(ACTION_BROADCAST_MASKS[ii])
    final_state = temp ^ (temp & pattern)
    return final_state & ((1 << 48) - 1)


def apply_transition_all_actions(states: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """
    Compute successors for all actions in a vectorized manner.

    Returns an array of shape (num_states, 256) where each row contains the
    successors of the corresponding state under all 256 actions.
    """
    temp = states[:, np.newaxis] ^ XFORM_MASK[np.newaxis, :]
    res = temp ^ (temp & ACTION_BROADCAST_MASKS[np.newaxis, :])
    return res.astype(np.uint64)


def _validate_gene_mac_s() -> None:
    """Sanity check that GENE_Mac_S has the expected structure and values."""
    exp = np.array(
        [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        ],
        dtype=np.int8,
    )
    if GENE_Mac_S.shape != (4, 2, 3, 2) or GENE_Mac_S.dtype != np.int8:
        raise RuntimeError("GENE_Mac_S structure invalid")
    if not np.array_equal(np.unique(GENE_Mac_S), np.array([-1, 1], dtype=np.int8)):
        raise RuntimeError("GENE_Mac_S values must be ±1")
    if not np.array_equal(GENE_Mac_S, exp):
        raise RuntimeError("GENE_Mac_S pattern mismatch")


def _roundtrip_sanity() -> None:
    """Verify that tensor_to_int and int_to_tensor define a consistent mapping."""
    T0 = GENE_Mac_S.copy()
    s = tensor_to_int(T0)
    T1 = int_to_tensor(s)
    if not np.array_equal(T0, T1):
        raise RuntimeError("tensor_to_int/int_to_tensor round-trip failed")


_validate_gene_mac_s()
_roundtrip_sanity()


