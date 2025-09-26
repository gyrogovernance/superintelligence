"""FROZEN channel definitions - Single source of truth for all channel mappings.

This module defines the immutable channel structure used throughout GyroSI:
- Global channel (all 48 bits)
- 8 Layer×Frame slabs (6 bits each: 3 rows × 2 cols at fixed layer, frame)
- Priority ordering for recovery ladder
- Bit index mapping formulas

FROZEN - These definitions are immutable and used across the entire system.
Validated at import time with comprehensive integrity checks.
"""

from typing import List, Tuple, Dict, Any
import sys
import hashlib

# === FROZEN CONSTANTS ===

# Total number of bits in the state representation
TOTAL_BITS = 48

# Number of slabs in the Layer×Frame structure
NUM_SLABS = 8

# Bits per slab (3 rows × 2 columns)
BITS_PER_SLAB = 6

# Tensor dimensions
NUM_LAYERS = 4
NUM_FRAMES = 2
NUM_ROWS = 3
NUM_COLS = 2

# === CHANNEL DEFINITIONS ===


class FROZEN_CHANNELS:
    """Immutable channel definitions for GyroSI physics."""

    # Constants from module level
    TOTAL_BITS = TOTAL_BITS
    NUM_SLABS = NUM_SLABS
    BITS_PER_SLAB = BITS_PER_SLAB
    NUM_LAYERS = NUM_LAYERS
    NUM_FRAMES = NUM_FRAMES
    NUM_ROWS = NUM_ROWS
    NUM_COLS = NUM_COLS

    # 48-bit mask for state representation
    MASK48 = 0xFFFFFFFFFFFF

    # Global channel: all 48 bit positions
    GLOBAL = list(range(TOTAL_BITS))

    # Layer×Frame slab definitions
    # Each slab contains 6 bits: 3 rows × 2 columns
    SLABS = {
        # Slab index: (layer, frame)
        0: (0, 0),  # Layer 0, Frame 0
        1: (0, 1),  # Layer 0, Frame 1
        2: (1, 0),  # Layer 1, Frame 0
        3: (1, 1),  # Layer 1, Frame 1
        4: (2, 0),  # Layer 2, Frame 0
        5: (2, 1),  # Layer 2, Frame 1
        6: (3, 0),  # Layer 3, Frame 0
        7: (3, 1),  # Layer 3, Frame 1
    }

    # Priority order for recovery ladder (Global always enabled)
    # Higher index = lower priority (dropped first in recovery)
    SLAB_PRIORITY_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]

    @staticmethod
    def get_bit_index(layer: int, frame: int, row: int, col: int) -> int:
        """Convert tensor coordinates to bit index.

        Formula: bit_index = (layer * 12) + (frame * 6) + (row * 2) + col

        Args:
            layer: Layer index (0-3)
            frame: Frame index (0-1)
            row: Row index (0-2)
            col: Column index (0-1)

        Returns:
            Bit index (0-47)
        """
        if not (0 <= layer < NUM_LAYERS):
            raise ValueError(f"Layer {layer} out of range [0, {NUM_LAYERS-1}]")
        if not (0 <= frame < NUM_FRAMES):
            raise ValueError(f"Frame {frame} out of range [0, {NUM_FRAMES-1}]")
        if not (0 <= row < NUM_ROWS):
            raise ValueError(f"Row {row} out of range [0, {NUM_ROWS-1}]")
        if not (0 <= col < NUM_COLS):
            raise ValueError(f"Column {col} out of range [0, {NUM_COLS-1}]")

        return (layer * 12) + (frame * 6) + (row * 2) + col

    @staticmethod
    def get_tensor_coords(bit_index: int) -> Tuple[int, int, int, int]:
        """Convert bit index to tensor coordinates.

        Args:
            bit_index: Bit index (0-47)

        Returns:
            Tuple of (layer, frame, row, col)
        """
        if not (0 <= bit_index < TOTAL_BITS):
            raise ValueError(f"Bit index {bit_index} out of range [0, {TOTAL_BITS-1}]")

        layer = bit_index // 12
        remainder = bit_index % 12
        frame = remainder // 6
        remainder = remainder % 6
        row = remainder // 2
        col = remainder % 2

        return (layer, frame, row, col)

    @staticmethod
    def get_slab_bit_indices(slab_idx: int) -> List[int]:
        """Get bit indices for a specific slab.

        Args:
            slab_idx: Slab index (0-7)

        Returns:
            List of 6 bit indices for the slab
        """
        if slab_idx not in FROZEN_CHANNELS.SLABS:
            raise ValueError(f"Invalid slab index {slab_idx}")

        layer, frame = FROZEN_CHANNELS.SLABS[slab_idx]
        indices = []

        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                bit_idx = FROZEN_CHANNELS.get_bit_index(layer, frame, row, col)
                indices.append(bit_idx)

        return indices

    @staticmethod
    def get_slab_mask(slab_idx: int) -> int:
        """Get bitmask for a specific slab.

        Args:
            slab_idx: Slab index (0-7)

        Returns:
            Bitmask with bits set for the slab positions
        """
        indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
        mask = 0
        for bit_idx in indices:
            mask |= 1 << bit_idx
        return mask

    @staticmethod
    def get_all_slab_masks() -> List[int]:
        """Get bitmasks for all slabs.

        Returns:
            List of 8 bitmasks, one for each slab
        """
        return [FROZEN_CHANNELS.get_slab_mask(i) for i in range(NUM_SLABS)]

    @staticmethod
    def verify_channel_integrity() -> None:
        """Comprehensive channel mapping integrity verification.

        Validates:
        - Slab structure and bit coverage
        - Coordinate mapping consistency
        - Priority ordering validity
        - Mathematical invariants
        - Cross-validation of all methods

        Raises:
            RuntimeError: If any integrity check fails
        """
        try:
            # 1. Verify dimensional constants
            assert TOTAL_BITS == 48, f"TOTAL_BITS must be 48, got {TOTAL_BITS}"
            assert NUM_SLABS == 8, f"NUM_SLABS must be 8, got {NUM_SLABS}"
            assert BITS_PER_SLAB == 6, f"BITS_PER_SLAB must be 6, got {BITS_PER_SLAB}"
            assert NUM_LAYERS == 4, f"NUM_LAYERS must be 4, got {NUM_LAYERS}"
            assert NUM_FRAMES == 2, f"NUM_FRAMES must be 2, got {NUM_FRAMES}"
            assert NUM_ROWS == 3, f"NUM_ROWS must be 3, got {NUM_ROWS}"
            assert NUM_COLS == 2, f"NUM_COLS must be 2, got {NUM_COLS}"

            # 2. Verify mathematical consistency
            assert NUM_LAYERS * NUM_FRAMES == NUM_SLABS, "Layer×Frame must equal NUM_SLABS"
            assert NUM_ROWS * NUM_COLS == BITS_PER_SLAB, "Rows×Cols must equal BITS_PER_SLAB"
            assert NUM_SLABS * BITS_PER_SLAB == TOTAL_BITS, "Slabs×BitsPerSlab must equal TOTAL_BITS"

            # 3. Verify each slab has exactly 6 distinct indices
            all_indices = set()
            for slab_idx in range(NUM_SLABS):
                slab_indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
                assert (
                    len(slab_indices) == BITS_PER_SLAB
                ), f"Slab {slab_idx} has {len(slab_indices)} indices, expected {BITS_PER_SLAB}"
                assert len(set(slab_indices)) == BITS_PER_SLAB, f"Slab {slab_idx} has duplicate indices: {slab_indices}"

                # Verify indices are in valid range
                for idx in slab_indices:
                    assert 0 <= idx < TOTAL_BITS, f"Slab {slab_idx} has out-of-range index {idx}"

                all_indices.update(slab_indices)

            # 4. Verify union of all slab indices covers exactly 0-47
            assert (
                len(all_indices) == TOTAL_BITS
            ), f"Union of all slab indices is {len(all_indices)}, expected {TOTAL_BITS}"
            assert all_indices == set(
                range(TOTAL_BITS)
            ), f"Slab indices do not cover exactly 0-{TOTAL_BITS-1}: {sorted(all_indices)}"

            # 5. Verify global channel covers all bits
            assert (
                len(FROZEN_CHANNELS.GLOBAL) == TOTAL_BITS
            ), f"Global channel has {len(FROZEN_CHANNELS.GLOBAL)} bits, expected {TOTAL_BITS}"
            assert set(FROZEN_CHANNELS.GLOBAL) == set(range(TOTAL_BITS)), "Global channel must cover exactly 0-47"

            # 6. Verify slab priority order
            assert (
                len(FROZEN_CHANNELS.SLAB_PRIORITY_ORDER) == NUM_SLABS
            ), f"Priority order has {len(FROZEN_CHANNELS.SLAB_PRIORITY_ORDER)} elements, expected {NUM_SLABS}"
            assert set(FROZEN_CHANNELS.SLAB_PRIORITY_ORDER) == set(
                range(NUM_SLABS)
            ), "Priority order must contain each slab index exactly once"

            # 7. Verify coordinate mapping consistency (round-trip test)
            for bit_idx in range(TOTAL_BITS):
                layer, frame, row, col = FROZEN_CHANNELS.get_tensor_coords(bit_idx)
                reconstructed_idx = FROZEN_CHANNELS.get_bit_index(layer, frame, row, col)
                assert (
                    reconstructed_idx == bit_idx
                ), f"Round-trip failed: {bit_idx} -> {(layer, frame, row, col)} -> {reconstructed_idx}"

            # 8. Verify slab definitions match coordinate mapping
            for slab_idx, (expected_layer, expected_frame) in FROZEN_CHANNELS.SLABS.items():
                slab_indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
                for bit_idx in slab_indices:
                    layer, frame, row, col = FROZEN_CHANNELS.get_tensor_coords(bit_idx)
                    assert (
                        layer == expected_layer
                    ), f"Slab {slab_idx} bit {bit_idx} has layer {layer}, expected {expected_layer}"
                    assert (
                        frame == expected_frame
                    ), f"Slab {slab_idx} bit {bit_idx} has frame {frame}, expected {expected_frame}"

            # 9. Verify mask generation consistency
            for slab_idx in range(NUM_SLABS):
                mask = FROZEN_CHANNELS.get_slab_mask(slab_idx)
                indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)

                # Check that mask has exactly the right bits set
                expected_mask = sum(1 << idx for idx in indices)
                assert (
                    mask == expected_mask
                ), f"Slab {slab_idx} mask mismatch: got {mask:048b}, expected {expected_mask:048b}"

                # Check popcount equals BITS_PER_SLAB
                assert (
                    bin(mask).count("1") == BITS_PER_SLAB
                ), f"Slab {slab_idx} mask has {bin(mask).count('1')} bits set, expected {BITS_PER_SLAB}"

            # 10. Verify MASK48 constant
            assert (
                FROZEN_CHANNELS.MASK48 == 0xFFFFFFFFFFFF
            ), f"MASK48 should be 0xFFFFFFFFFFFF, got {FROZEN_CHANNELS.MASK48:012x}"
            assert FROZEN_CHANNELS.MASK48 == (1 << TOTAL_BITS) - 1, "MASK48 should equal (1 << 48) - 1"

        except Exception as e:
            raise RuntimeError(f"Channel integrity check failed: {e}") from e

    @staticmethod
    def get_definition_hash() -> str:
        """Generate cryptographic hash of channel definitions.

        Returns:
            SHA-256 hash of all channel definitions for integrity verification
        """
        # Collect all critical definitions in Traceable order
        definition_data = {
            "constants": {
                "TOTAL_BITS": TOTAL_BITS,
                "NUM_SLABS": NUM_SLABS,
                "BITS_PER_SLAB": BITS_PER_SLAB,
                "NUM_LAYERS": NUM_LAYERS,
                "NUM_FRAMES": NUM_FRAMES,
                "NUM_ROWS": NUM_ROWS,
                "NUM_COLS": NUM_COLS,
                "MASK48": FROZEN_CHANNELS.MASK48,
            },
            "slabs": dict(sorted(FROZEN_CHANNELS.SLABS.items())),
            "priority_order": FROZEN_CHANNELS.SLAB_PRIORITY_ORDER,
            "global_channel": FROZEN_CHANNELS.GLOBAL,
        }

        # Convert to Traceable string representation
        definition_str = str(sorted(definition_data.items()))

        # Generate SHA-256 hash
        return hashlib.sha256(definition_str.encode("utf-8")).hexdigest()

    @staticmethod
    def validate_import() -> None:
        """Comprehensive validation performed at import time.

        Validates channel integrity and logs validation status.
        Exits with error code if validation fails.
        """
        try:
            # Run comprehensive integrity check
            FROZEN_CHANNELS.verify_channel_integrity()

            # Log successful validation (only in debug mode to avoid spam)
            if __debug__:
                print(f"[FROZEN_CHANNELS] Integrity validated successfully.", file=sys.stderr)

        except Exception as e:
            print(f"[FROZEN_CHANNELS] CRITICAL: Channel integrity validation failed: {e}", file=sys.stderr)
            print(
                f"[FROZEN_CHANNELS] This indicates corrupted channel definitions. System cannot continue.",
                file=sys.stderr,
            )
            sys.exit(1)


# === CONVENIENCE FUNCTIONS ===


def get_slab_name(slab_idx: int) -> str:
    """Get human-readable name for a slab.

    Args:
        slab_idx: Slab index (0-7)

    Returns:
        String like "Layer×Frame[0,0]"
    """
    if slab_idx not in FROZEN_CHANNELS.SLABS:
        raise ValueError(f"Invalid slab index {slab_idx}")

    layer, frame = FROZEN_CHANNELS.SLABS[slab_idx]
    return f"Layer×Frame[{layer},{frame}]"


def get_channel_summary() -> Dict[str, Any]:
    """Get summary of all channel definitions.

    Returns:
        Dictionary with channel information
    """
    return {
        "total_bits": TOTAL_BITS,
        "num_slabs": NUM_SLABS,
        "bits_per_slab": BITS_PER_SLAB,
        "global_channel_size": len(FROZEN_CHANNELS.GLOBAL),
        "slab_names": [get_slab_name(i) for i in range(NUM_SLABS)],
        "priority_order": [get_slab_name(i) for i in FROZEN_CHANNELS.SLAB_PRIORITY_ORDER],
    }


# === IMPORT-TIME VALIDATION ===

# Verify integrity on import with comprehensive checks
FROZEN_CHANNELS.validate_import()
