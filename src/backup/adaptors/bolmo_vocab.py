"""
Bolmo token/byte mapping helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BolmoVocabSpec:
    offset: int = 4
    base_size: int = 256

    @property
    def boundary_offset(self) -> int:
        return self.offset + self.base_size

    @property
    def base_start(self) -> int:
        return self.offset

    @property
    def base_end_exclusive(self) -> int:
        return self.offset + self.base_size

    @property
    def fused_start(self) -> int:
        return self.boundary_offset

    @property
    def fused_end_exclusive(self) -> int:
        return self.boundary_offset + self.base_size


def token_to_byte_and_fused(token_id: int, spec: BolmoVocabSpec) -> tuple[int | None, bool]:
    tid = int(token_id)
    if tid < spec.offset:
        return None, False
    if spec.base_start <= tid < spec.base_end_exclusive:
        return tid - spec.offset, False
    if spec.fused_start <= tid < spec.fused_end_exclusive:
        return tid - spec.boundary_offset, True
    return None, False


def byte_to_base_token(byte_val: int, spec: BolmoVocabSpec) -> int:
    b = int(byte_val) & 0xFF
    return spec.offset + b
