from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .ops import (
    gyrograph_word_signature_from_bytes,
)


@dataclass(frozen=True)
class ByteOp:
    payload: int
    family: int


@dataclass(frozen=True)
class GateOp:
    gate: str


@dataclass(frozen=True)
class WHTOp:
    pass


@dataclass(frozen=True)
class ObserveOp:
    observable: str


@dataclass(frozen=True)
class ConditionOp:
    observable: str
    predicate: Callable[[int], bool]
    then_ops: Sequence[object]
    else_ops: Sequence[object]


@dataclass(frozen=True)
class SubCircuit:
    name: str
    ops: Sequence[object]


def optimize_bytes(byte_seq: bytes) -> bytes:
    """
    SDK v1 conformant behavior: preserve exact byte sequence.
    No peephole rewrite is performed unless a real shortening rule exists.
    """
    return byte_seq if isinstance(byte_seq, bytes) else bytes(byte_seq)


def compile_circuit(
    ops: Sequence[object],
) -> tuple[bytes, object, list]:
    bytes_out: list[int] = []
    ir: list = [op for op in ops if not isinstance(op, (ByteOp, GateOp))]
    for op in ops:
        if isinstance(op, ByteOp):
            pl = int(op.payload) & 0x3F
            fam = int(op.family) & 3
            intron = ((fam >> 1) << 7) | (pl << 1) | (fam & 1)
            bytes_out.append((intron ^ 0xAA) & 0xFF)
        elif isinstance(op, GateOp):
            g = op.gate
            if g == "S":
                bytes_out.extend([0xAA, 0x54])
            elif g == "C":
                bytes_out.extend([0xD5, 0x2B])
            elif g == "F":
                bytes_out.extend([0xAA, 0x54, 0xD5, 0x2B])
            elif g == "id":
                pass
            else:
                raise ValueError(f"unknown gate: {g!r}")
        elif isinstance(op, SubCircuit):
            inner_b, _, _ = compile_circuit(tuple(op.ops))
            bytes_out.extend(inner_b)
        elif isinstance(op, (WHTOp, ObserveOp, ConditionOp)):
            continue
        else:
            raise TypeError(f"unsupported op: {type(op)!r}")
    out_b = optimize_bytes(bytes(bytes_out))
    sig = gyrograph_word_signature_from_bytes(out_b)
    return out_b, sig, ir


__all__ = [
    "ByteOp",
    "ConditionOp",
    "GateOp",
    "ObserveOp",
    "SubCircuit",
    "WHTOp",
    "compile_circuit",
    "optimize_bytes",
]
