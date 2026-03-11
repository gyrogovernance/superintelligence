"""
Coordinator: Kernel + App ledgers + tools.

Responsibilities:
- advance kernel state (shared moment) by bytes
- accept governance events from tools/app
- update domain ledgers deterministically
- expose GGG apertures (ledger-based), plus kernel signature
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore

from src.kernel import Gyroscopic

from .events import MICRO, Domain, EdgeID, GovernanceEvent, Grant, Shell
from .ledger import DomainLedgers


@dataclass
class CoordinationStatus:
    kernel: dict[str, Any]
    ledgers: dict[str, Any]
    apertures: dict[str, float]
    fiat: dict[str, Any] | None = None


# CSM (Common Source Moment) capacity constants
# SI second definition: caesium-133 hyperfine transition frequency
ATOMIC_HZ_CS133: int = 9_192_631_770

# Router reachable shared-moment space (BFS-verified from rest state)
OMEGA_SIZE: int = 4_096


def raw_microcells_per_moment() -> float:
    """
    N_phys = (4/3)pi f_Cs^3

    Raw physical microcells in 1-second light-sphere volume at atomic
    wavelength resolution. The speed of light cancels exactly.
    """
    f = ATOMIC_HZ_CS133
    return (4.0 / 3.0) * math.pi * (f ** 3)


def csm_total_mu() -> float:
    """
    CSM = N_phys / |Omega|

    Total Common Source Moment capacity in MU.
    |Omega| = 4,096 (BFS-verified reachable shared-moment space).
    """
    return raw_microcells_per_moment() / float(OMEGA_SIZE)


class Coordinator:
    def __init__(self) -> None:
        self.kernel = Gyroscopic()
        self.ledgers = DomainLedgers()

        # Audit logs
        self.byte_log: list[int] = []
        self.event_log: list[dict[str, Any]] = []

        # Settlement medium state
        self.sealer = Gyroscopic()
        self.fiat_grants_current: dict[str, Grant] = {}  # key = identity_id
        self.fiat_shell_log: list[Shell] = []
        self.fiat_shell_grants_log: list[tuple[Shell, list[Grant]]] = []
        self.fiat_archive_totals: dict[str, int] = {}  # key = identity_id
        self.fiat_identity_id_to_identity: dict[str, str] = {}
        self.fiat_used_total: int = 0
        self.fiat_capacity_total: int = 0

    # -------------------------
    # Kernel stepping (shared moment)
    # -------------------------
    def step_byte(self, byte: int, emit_system_event: bool = True) -> None:
        """
        Step the kernel by one byte.

        If emit_system_event is True, emits a small Economy domain event
        representing the structural change.
        """
        b = int(byte) & 0xFF
        self.kernel.step_byte(b)
        self.byte_log.append(b)

        if emit_system_event:
            system_event = GovernanceEvent(
                domain=Domain.ECONOMY,
                edge_id=EdgeID.GOV_INFO,
                magnitude_micro=int(0.01 * MICRO),
                confidence_micro=MICRO,
                meta={"type": "kernel_step", "byte": b},
            )
            self.apply_event(system_event, bind_to_kernel_moment=True)

    def step_bytes(self, payload: bytes) -> None:
        for b in payload:
            self.step_byte(b)

    # -------------------------
    # App events (ledger updates)
    # -------------------------
    def apply_event(self, ev: GovernanceEvent, bind_to_kernel_moment: bool = True) -> None:
        """
        Apply governance event to the appropriate domain ledger.
        Optionally bind it to the current kernel shared moment for replay.
        """
        if bind_to_kernel_moment:
            ev = GovernanceEvent(
                domain=ev.domain,
                edge_id=ev.edge_id,
                magnitude_micro=ev.magnitude_micro,
                confidence_micro=ev.confidence_micro,
                meta=dict(ev.meta),
                kernel_step=self.kernel.step,
                kernel_state24=self.kernel.state24,
                kernel_last_byte=int(self.kernel.last_byte),
            )

        self.ledgers.apply_event(ev)

        self.event_log.append(
            {
                "event_index": len(self.event_log),
                "kernel_step": ev.kernel_step,
                "kernel_state24": ev.kernel_state24,
                "kernel_last_byte": ev.kernel_last_byte,
                "event": ev.as_dict(),
            }
        )

    # -------------------------
    # Reporting
    # -------------------------
    def get_status(self) -> CoordinationStatus:
        sig = self.kernel.signature()

        kernel_info = {
            "step": sig.step,
            "state24": sig.state24,
            "state_hex": sig.state_hex,
            "a_hex": sig.a_hex,
            "b_hex": sig.b_hex,
            "last_byte": int(self.kernel.last_byte),
            "byte_log_len": len(self.byte_log),
            "event_log_len": len(self.event_log),
        }

        apertures = {
            "econ": self.ledgers.aperture(Domain.ECONOMY),
            "emp": self.ledgers.aperture(Domain.EMPLOYMENT),
            "edu": self.ledgers.aperture(Domain.EDUCATION),
        }

        ledgers = {
            "y_econ": self.ledgers.get(Domain.ECONOMY).tolist(),
            "y_emp": self.ledgers.get(Domain.EMPLOYMENT).tolist(),
            "y_edu": self.ledgers.get(Domain.EDUCATION).tolist(),
            "event_count": self.ledgers.event_count,
        }

        fiat_info = self.fiat_status()

        return CoordinationStatus(kernel=kernel_info, ledgers=ledgers, apertures=apertures, fiat=fiat_info)

    def anchor_identity(self, name: str) -> tuple[str, str]:
        """
        Compute identity_id and anchor for an identity name.

        Returns:
            (identity_id, anchor) where:
            - identity_id: SHA-256 hex of identity seed (64 hex chars)
            - anchor: Router state_hex (6 hex chars)
        """
        seed = f"identity:{name}".encode()
        h = hashlib.sha256(seed).digest()
        identity_id = h.hex()
        sig = self.sealer.route_from_archetype(h)
        anchor = sig.state_hex
        return identity_id, anchor

    def add_grant(self, identity: str, mu_allocated: int, header: bytes | str | None = None) -> None:
        """
        Add a grant to the current shell.

        Raises:
            ValueError: if mu_allocated < 0 or identity already has a grant
        """
        if mu_allocated < 0:
            raise ValueError(f"mu_allocated must be non-negative, got {mu_allocated}")

        identity_id, anchor = self.anchor_identity(identity)
        if identity_id in self.fiat_grants_current:
            raise ValueError("Identity already has a grant in current shell")

        grant = Grant(identity=identity, identity_id=identity_id, anchor=anchor, mu_allocated=mu_allocated)
        self.fiat_grants_current[identity_id] = grant

    def close_shell(self, header: bytes | str, total_capacity_MU: int) -> Shell:
        """
        Close the current shell by building receipts and sealing.

        Returns:
            Shell object with seal and capacity metrics

        Raises:
            ValueError: if used capacity exceeds total capacity
        """
        if isinstance(header, bytes):
            header_str = header.decode("utf-8", errors="replace")
            header_bytes = header
        else:
            header_str = str(header)
            header_bytes = header_str.encode("utf-8")

        grants_list = list(self.fiat_grants_current.values())
        receipts: list[bytes] = []
        for g in sorted(grants_list, key=lambda gg: gg.identity_id):
            mu_bytes = g.mu_allocated.to_bytes(8, "big", signed=False)
            receipt = bytes.fromhex(g.identity_id) + bytes.fromhex(g.anchor) + mu_bytes
            receipts.append(receipt)

        payload = header_bytes + b"".join(receipts)
        sig = self.sealer.route_from_archetype(payload)
        seal = sig.state_hex

        used = sum(g.mu_allocated for g in grants_list)
        free = total_capacity_MU - used

        if used > total_capacity_MU:
            raise ValueError(f"Used capacity {used} exceeds total capacity {total_capacity_MU}")

        shell = Shell(
            header=header_str,
            seal=seal,
            total_capacity_MU=total_capacity_MU,
            used_capacity_MU=used,
            free_capacity_MU=free,
        )

        self.fiat_shell_log.append(shell)
        self.fiat_shell_grants_log.append((shell, grants_list))
        self.fiat_capacity_total += total_capacity_MU
        self.fiat_used_total += used
        for g in grants_list:
            self.fiat_archive_totals[g.identity_id] = self.fiat_archive_totals.get(g.identity_id, 0) + g.mu_allocated
            self.fiat_identity_id_to_identity[g.identity_id] = g.identity

        self.fiat_grants_current.clear()

        return shell

    def fiat_status(self) -> dict[str, Any]:
        """Return settlement medium status."""
        per_identity_totals: dict[str, int] = {}
        for identity_id, total in self.fiat_archive_totals.items():
            identity = self.fiat_identity_id_to_identity.get(identity_id, identity_id)
            per_identity_totals[identity] = per_identity_totals.get(identity, 0) + total

        status: dict[str, Any] = {
            "pending_grants_count": len(self.fiat_grants_current),
            "total_capacity_MU": self.fiat_capacity_total,
            "used_capacity_MU": self.fiat_used_total,
            "free_capacity_MU": self.fiat_capacity_total - self.fiat_used_total,
            "shell_count": len(self.fiat_shell_log),
            "per_anchor_totals": dict(self.fiat_archive_totals),
            "per_identity_totals": per_identity_totals,
        }

        if self.fiat_shell_log:
            last_shell = self.fiat_shell_log[-1]
            status["current_shell_header"] = last_shell.header
            status["last_shell_seal"] = last_shell.seal
            seals = [shell.seal for shell in self.fiat_shell_log]
            status["meta_root"] = self.meta_root_from_shell_seals(seals)
        else:
            status["current_shell_header"] = None
            status["last_shell_seal"] = None
            status["meta_root"] = None

        return status

    def meta_root_from_shell_seals(self, seals: list[str]) -> str:
        """Compute meta-root from list of shell seals (sorted)."""
        seals_sorted = sorted(seals)
        seal_bytes = [bytes.fromhex(s) for s in seals_sorted]
        sig = self.sealer.route_from_archetype(b"".join(seal_bytes))
        return sig.state_hex

    def rollback_last_shell(self) -> None:
        """Rollback the last shell: remove it and reverse archive totals."""
        if not self.fiat_shell_grants_log:
            raise ValueError("No shells to rollback")

        shell, grants = self.fiat_shell_grants_log.pop()

        assert self.fiat_shell_log and self.fiat_shell_log[-1].seal == shell.seal
        self.fiat_shell_log.pop()

        self.fiat_capacity_total -= shell.total_capacity_MU
        self.fiat_used_total -= shell.used_capacity_MU

        for g in grants:
            prev = self.fiat_archive_totals.get(g.identity_id, 0)
            new = prev - g.mu_allocated
            if new <= 0:
                self.fiat_archive_totals.pop(g.identity_id, None)
                self.fiat_identity_id_to_identity.pop(g.identity_id, None)
            else:
                self.fiat_archive_totals[g.identity_id] = new

    def rollback_kernel_steps(self, n: int) -> None:
        """Rollback n kernel steps using inverse stepping."""
        from src.constants import GENE_MIC_S

        n = min(n, self.kernel.step)
        for _ in range(n):
            if not self.byte_log:
                break
            b = self.byte_log.pop()
            self.kernel.step_byte_inverse(b)
        self.kernel.last_byte = self.byte_log[-1] if self.byte_log else GENE_MIC_S
        assert self.kernel.step == len(self.byte_log)

    def sign_bundle(self, bundle_json_bytes: bytes, private_key: Any) -> bytes:
        """Sign bundle JSON bytes using Ed25519."""
        if Ed25519PrivateKey is None:
            raise ImportError("cryptography package required for bundle signing")
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"private_key must be Ed25519PrivateKey, got {type(private_key)}")

        digest = hashlib.sha256(bundle_json_bytes).digest()
        return private_key.sign(digest)

    def verify_bundle_signature(self, bundle_json_bytes: bytes, signature: bytes, public_key: Any) -> bool:
        """Verify bundle JSON bytes signature using Ed25519."""
        if Ed25519PublicKey is None:
            raise ImportError("cryptography package required for bundle signature verification")
        if not isinstance(public_key, Ed25519PublicKey):
            raise TypeError(f"public_key must be Ed25519PublicKey, got {type(public_key)}")

        try:
            digest = hashlib.sha256(bundle_json_bytes).digest()
            public_key.verify(signature, digest)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self.kernel.reset()
        self.ledgers = DomainLedgers()
        self.byte_log.clear()
        self.event_log.clear()

        self.sealer = Gyroscopic()
        self.fiat_grants_current.clear()
        self.fiat_shell_log.clear()
        self.fiat_shell_grants_log.clear()
        self.fiat_archive_totals.clear()
        self.fiat_identity_id_to_identity.clear()
        self.fiat_used_total = 0
        self.fiat_capacity_total = 0

    def derive_domain_counts(self) -> dict[str, int]:
        """Derive domain_counts from the event log."""
        counts = {
            "economy": 0,
            "employment": 0,
            "education": 0,
        }

        for log_entry in self.event_log:
            event_dict = log_entry.get("event", {})
            domain_int = event_dict.get("domain")
            if domain_int is not None:
                domain = Domain(domain_int)
                if domain == Domain.ECONOMY:
                    counts["economy"] += 1
                elif domain == Domain.EMPLOYMENT:
                    counts["employment"] += 1
                elif domain == Domain.EDUCATION:
                    counts["education"] += 1

        return counts