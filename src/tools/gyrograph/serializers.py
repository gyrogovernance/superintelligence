from __future__ import annotations


def pack_word4(b0: int, b1: int, b2: int, b3: int) -> bytes:
    """Pack four bytes into the native exact 4-byte GyroGraph word."""
    return bytes((b0 & 0xFF, b1 & 0xFF, b2 & 0xFF, b3 & 0xFF))


def ensure_word4(word: bytes | bytearray | memoryview) -> bytes:
    """
    Validate that the given object is exactly one closed 4-byte word.

    No padding, chunking, or bridge policy lives here.
    """
    w = bytes(word)
    if len(w) != 4:
        raise ValueError(f"Expected exact 4-byte word, got length {len(w)}")
    return w