"""
Router Ledger

Implements a minimal append-only binary ledger for Router kernel events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional


EVENT_EGRESS = 0x01
RECORD_SIZE = 14  # event_code (1) + state_before (6) + action_byte (1) + state_after (6)


@dataclass
class Ledger:
    """
    Append-only binary ledger for Router kernel events.

    Each record has fixed width and encodes:
    - event_code: uint8
    - state_before: 6-byte big-endian integer
    - action_byte: uint8
    - state_after: 6-byte big-endian integer
    """

    path: Path
    _file: Optional[BinaryIO] = None

    def open(self) -> None:
        """Open the ledger file for appending."""
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("ab")

    def close(self) -> None:
        """Close the ledger file if it is open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def append_egress(
        self,
        state_before: int,
        action_byte: int,
        state_after: int,
    ) -> None:
        """
        Append a single BU-egress event to the ledger.

        All integer fields are stored in fixed-width big-endian format to make
        replay and auditing deterministic.
        """
        if self._file is None:
            self.open()

        event_code = EVENT_EGRESS.to_bytes(1, "big")
        sb = int(state_before).to_bytes(6, "big")
        action_b = (int(action_byte) & 0xFF).to_bytes(1, "big")
        sa = int(state_after).to_bytes(6, "big")

        record = event_code + sb + action_b + sa
        if len(record) != RECORD_SIZE:
            raise ValueError("ledger record has incorrect size")

        assert self._file is not None
        self._file.write(record)
        self._file.flush()


