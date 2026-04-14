"""
SDK 10.2 target profiles and Python vs native moment conformance.

TargetProfile.native_ops is a frozenset of op name strings (immutable set).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import GENE_MAC_REST
from src.sdk import moment_from_ledger


@dataclass(frozen=True)
class TargetProfile:
    name: str
    native_ops: frozenset[str]
    step_semantics: str
    state_inspection: str
    provenance_format: str
    wht_support: str


PYTHON_KERNEL = TargetProfile(
    "PythonKernel",
    frozenset({"byte", "gate", "wht", "observe"}),
    "v1",
    "full",
    "ledger",
    "matrix",
)

CENGINE = TargetProfile(
    "CEngine",
    frozenset({"byte", "gate", "wht", "observe"}),
    "v1",
    "full",
    "ledger",
    "native",
)


def test_target_equivalence(ledger: bytes) -> bool:
    from .ops import gyrograph_moment_from_ledger_native

    py_m = moment_from_ledger(ledger, start_state24=GENE_MAC_REST)
    c_m = gyrograph_moment_from_ledger_native(ledger)
    return (
        int(py_m.state24) == int(c_m.state24)
        and int(py_m.step) == int(c_m.step)
        and int(py_m.last_byte) == int(c_m.last_byte)
    )


__all__ = [
    "CENGINE",
    "PYTHON_KERNEL",
    "TargetProfile",
    "test_target_equivalence",
]
