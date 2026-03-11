"""
Shared helpers for Moments Economy and Genealogy tests.

Identity anchors, Grant, Shell, and _make_shell live here so both
test_moments_economy and test_moments_genealogy import from one place.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROGRAM_ROOT = Path(__file__).parent.parent
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from src.kernel import Gyroscopic


def identity_hash(name: str) -> bytes:
    """Collision-resistant hash of identity string."""
    return hashlib.sha256(name.encode("utf-8")).digest()


def identity_anchor(name: str) -> tuple[bytes, str]:
    """
    Identity Anchor: (identifier, kernel_anchor).
    Identifier = SHA-256 of name.
    Kernel anchor = Router state_hex after routing identifier from rest.
    """
    ident = identity_hash(name)
    r = Gyroscopic()
    sig = r.route_from_archetype(ident)
    return ident, sig.state_hex


@dataclass(frozen=True)
class Grant:
    """A single MU allocation to an identity."""
    identity_label: str
    identity_id: bytes
    kernel_anchor: str
    amount_mu: int

    def canonical_receipt(self) -> bytes:
        """Canonical byte representation for seal computation."""
        return self.identity_id + self.kernel_anchor.encode("ascii") + \
            self.amount_mu.to_bytes(8, "big")


@dataclass
class Shell:
    """Time-bounded capacity container with deterministic seal."""
    header: bytes
    grants: list[Grant] = field(default_factory=list)
    total_capacity_mu: int = 0
    seal: str = ""

    @property
    def used_capacity_mu(self) -> int:
        return sum(g.amount_mu for g in self.grants)

    @property
    def free_capacity_mu(self) -> int:
        return self.total_capacity_mu - self.used_capacity_mu

    def compute_seal(self) -> str:
        """
        Seal = Router state after routing canonical byte sequence from rest.
        1. Sort grants by identity_id.
        2. Concatenate header + sorted canonical receipts.
        3. Route from archetype.
        4. Record state_hex.
        """
        sorted_grants = sorted(self.grants, key=lambda g: g.identity_id)
        payload = self.header
        for g in sorted_grants:
            payload += g.canonical_receipt()
        r = Gyroscopic()
        sig = r.route_from_archetype(payload)
        self.seal = sig.state_hex
        return self.seal


def _make_shell(header: bytes, grants_spec: list[tuple[str, int]],
                capacity: int = 10**18) -> Shell:
    """Helper: build a Shell from (name, amount) pairs."""
    grants = []
    for name, amount in grants_spec:
        ident, anchor = identity_anchor(name)
        grants.append(Grant(name, ident, anchor, amount))
    shell = Shell(header=header, grants=grants, total_capacity_mu=capacity)
    shell.compute_seal()
    return shell
