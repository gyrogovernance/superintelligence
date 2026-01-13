"""
Coordinator: Kernel + App ledgers + plugins.

This is the "spine" that a future UI (React, CLI, etc.) will call.

Responsibilities:
- advance kernel state (shared moment) by bytes
- accept governance events from plugins/app
- update domain ledgers deterministically
- expose GGG apertures (ledger-based), plus kernel signature
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey  # type: ignore
except ImportError:
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore

from src.router.kernel import RouterKernel

from .events import GovernanceEvent, Domain, EdgeID, Grant, Shell, MICRO
from .ledger import DomainLedgers


@dataclass
class CoordinationStatus:
    kernel: Dict[str, Any]
    ledgers: Dict[str, Any]
    apertures: Dict[str, float]
    fiat: Dict[str, Any] | None = None


# CSM (Common Source Moment) capacity constants
# These are the canonical capacity standard derived from physics + Router structure
# Proven in tests/test_moments_2.py

# SI second definition: caesium-133 hyperfine transition frequency
ATOMIC_HZ_CS133: int = 9_192_631_770

# Router ontology size (proven as C × C where |C| = 256)
OMEGA_SIZE: int = 65_536


def raw_microcells_per_moment() -> float:
    """
    N_phys = (4/3)π f_Cs³
    
    Raw physical microcells in 1-second light-sphere volume at atomic wavelength resolution.
    The speed of light cancels exactly (proven in test_moments_2.py).
    """
    f = ATOMIC_HZ_CS133
    return (4.0 / 3.0) * math.pi * (f ** 3)


def csm_total_mu() -> float:
    """
    CSM = N_phys / |Ω|
    
    Total Common Source Moment capacity in MU.
    
    This is the TOTAL structural capacity, derived from the volume of a 1-second 
    light-sphere at atomic resolution (N_phys), coarse-grained by the Router 
    ontology size (|Ω| = 65,536).
    
    The "1 second" is consumed in the derivation of N_phys; CSM is not a rate 
    and does not accumulate over time. It is a fixed total capacity.
    
    The uniform division by |Ω| is forced by symmetry: the Router's transitive 
    group action plus physical isotropy of the light-sphere require uniform 
    coarse-graining. This is the unique symmetry-invariant measure (proven in 
    test_moments_2.py).
    """
    return raw_microcells_per_moment() / float(OMEGA_SIZE)




class Coordinator:
    def __init__(self, atlas_dir: Path) -> None:
        self.atlas_dir = atlas_dir
        self.kernel = RouterKernel(atlas_dir)
        self.ledgers = DomainLedgers()

        # Audit logs (kept simple; you can persist externally)
        self.byte_log: List[int] = []
        self.event_log: List[Dict[str, Any]] = []

        # Fiat substrate state
        self.sealer = RouterKernel(atlas_dir)
        self.fiat_grants_current: Dict[str, Grant] = {}  # key = identity_id
        self.fiat_shell_log: List[Shell] = []
        self.fiat_shell_grants_log: List[Tuple[Shell, List[Grant]]] = []
        self.fiat_archive_totals: Dict[str, int] = {}  # key = identity_id (collision-resistant)
        self.fiat_identity_id_to_identity: Dict[str, str] = {}  # identity_id -> last seen identity label (for reporting)
        self.fiat_used_total: int = 0
        self.fiat_capacity_total: int = 0

    # -------------------------
    # Kernel stepping (shared moment)
    # -------------------------
    def step_byte(self, byte: int, emit_system_event: bool = True) -> None:
        """
        Step the kernel by one byte.
        
        Per GGG hierarchy: Kernel = Economy domain (structural substrate).
        If emit_system_event is True, emits a small Economy domain event
        representing the structural change.
        """
        b = int(byte) & 0xFF
        self.kernel.step_byte(b)
        self.byte_log.append(b)
        
        # Emit Economy domain event for kernel structural activity
        if emit_system_event:
            # Small magnitude representing structural substrate evolution
            # Using GOV_INFO edge as primary structural coupling
            system_event = GovernanceEvent(
                domain=Domain.ECONOMY,
                edge_id=EdgeID.GOV_INFO,
                magnitude_micro=int(0.01 * MICRO),  # Small structural change per byte
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
        Optionally bind it to the current kernel moment for replay.
        """
        if bind_to_kernel_moment:
            ev = GovernanceEvent(
                domain=ev.domain,
                edge_id=ev.edge_id,
                magnitude_micro=ev.magnitude_micro,
                confidence_micro=ev.confidence_micro,
                meta=dict(ev.meta),  # Copy dict to preserve audit trail immutability
                kernel_step=self.kernel.step,
                kernel_state_index=self.kernel.state_index,
                kernel_last_byte=self.kernel.last_byte,
            )

        self.ledgers.apply_event(ev)

        self.event_log.append(
            {
                "event_index": len(self.event_log),
                "kernel_step": ev.kernel_step,
                "kernel_state_index": ev.kernel_state_index,
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
            "state_index": sig.state_index,
            "state_hex": sig.state_hex,
            "a_hex": sig.a_hex,
            "b_hex": sig.b_hex,
            "last_byte": self.kernel.last_byte,
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

    def anchor_identity(self, name: str) -> Tuple[str, str]:
        """
        Compute identity_id and anchor for an identity name.
        
        Returns:
            (identity_id, anchor) tuple where:
            - identity_id: SHA-256 hex of identity seed (64 hex chars, collision-resistant)
            - anchor: Router state_hex (6 hex chars, structural coordinate)
        
        The identity_id provides collision resistance (256 bits).
        The anchor provides structural meaning in kernel phase space.
        """
        seed = f"identity:{name}".encode("utf-8")
        h = hashlib.sha256(seed).digest()
        identity_id = h.hex()  # Full 256-bit hash (64 hex chars)
        sig = self.sealer.route_from_archetype(h)
        anchor = sig.state_hex  # Structural coordinate (6 hex chars)
        return identity_id, anchor

    def add_grant(self, identity: str, mu_allocated: int, header: bytes | str | None = None) -> None:
        """
        Add a grant to the current shell.
        
        Args:
            identity: human-readable label
            mu_allocated: MU allocated to this identity
            header: optional header (unused for grant, kept for API compatibility)
        
        Raises:
            ValueError: if mu_allocated < 0 or identity already has a grant
        """
        if mu_allocated < 0:
            raise ValueError(f"mu_allocated must be non-negative, got {mu_allocated}")
        
        identity_id, anchor = self.anchor_identity(identity)
        if identity_id in self.fiat_grants_current:
            raise ValueError(f"Identity already has a grant in current shell")
        
        grant = Grant(identity=identity, identity_id=identity_id, anchor=anchor, mu_allocated=mu_allocated)
        self.fiat_grants_current[identity_id] = grant

    def close_shell(self, header: bytes | str, total_capacity_MU: int) -> Shell:
        """
        Close the current shell by building receipts and sealing.
        
        Args:
            header: contextual label (e.g. b"ecology:year:2026")
            total_capacity_MU: total MU capacity for this shell (for accounting purposes)
        
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
        
        # Build receipts by sorting grants by identity_id (canonical ordering)
        # Receipt = identity_id (32 bytes) || anchor (3 bytes) || mu (8 bytes)
        grants_list = list(self.fiat_grants_current.values())
        receipts: List[bytes] = []
        for g in sorted(grants_list, key=lambda gg: gg.identity_id):
            mu_bytes = g.mu_allocated.to_bytes(8, "big", signed=False)
            receipt = bytes.fromhex(g.identity_id) + bytes.fromhex(g.anchor) + mu_bytes
            receipts.append(receipt)
        
        # Route header || receipts through sealer
        payload = header_bytes + b"".join(receipts)
        sig = self.sealer.route_from_archetype(payload)
        seal = sig.state_hex
        
        # Compute used/free capacity
        used = sum(g.mu_allocated for g in grants_list)
        free = total_capacity_MU - used
        
        if used > total_capacity_MU:
            raise ValueError(f"Used capacity {used} exceeds total capacity {total_capacity_MU}")
        
        # Create shell
        shell = Shell(
            header=header_str,
            seal=seal,
            total_capacity_MU=total_capacity_MU,
            used_capacity_MU=used,
            free_capacity_MU=free,
        )
        
        # Update state
        self.fiat_shell_log.append(shell)
        self.fiat_shell_grants_log.append((shell, grants_list))
        self.fiat_capacity_total += total_capacity_MU
        self.fiat_used_total += used
        for g in grants_list:
            self.fiat_archive_totals[g.identity_id] = self.fiat_archive_totals.get(g.identity_id, 0) + g.mu_allocated
            self.fiat_identity_id_to_identity[g.identity_id] = g.identity
        
        # Clear current grants
        self.fiat_grants_current.clear()
        
        return shell

    def fiat_status(self) -> Dict[str, Any]:
        """
        Return fiat substrate status.
        
        Returns dict with:
        - current_shell_header: header of last shell (if any)
        - pending_grants_count: number of grants in current shell
        - last_shell_seal: seal of last shell (if any)
        - total_capacity_MU: sum of all shell capacities
        - used_capacity_MU: sum of all used capacities
        - free_capacity_MU: remaining capacity
        - shell_count: number of closed shells
        - per_identity_totals: per-identity MU totals (maybe truncated)
        - meta_root: optional meta-root from shell seals
        """
        # Derive per-identity totals from identity_id totals (for reporting)
        per_identity_totals: Dict[str, int] = {}
        for identity_id, total in self.fiat_archive_totals.items():
            identity = self.fiat_identity_id_to_identity.get(identity_id, identity_id)
            per_identity_totals[identity] = per_identity_totals.get(identity, 0) + total
        
        status: Dict[str, Any] = {
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

    def meta_root_from_shell_seals(self, seals: List[str]) -> str:
        """
        Compute meta-root from list of shell seals.
        
        Args:
            seals: list of seal hex strings
        
        Returns:
            meta-root state_hex (canonical ordering: seals are sorted)
        """
        seals_sorted = sorted(seals)
        seal_bytes = [bytes.fromhex(s) for s in seals_sorted]
        sig = self.sealer.route_from_archetype(b"".join(seal_bytes))
        return sig.state_hex

    def rollback_last_shell(self) -> None:
        """
        Rollback the last shell: remove it and reverse archive totals.
        """
        if not self.fiat_shell_grants_log:
            raise ValueError("No shells to rollback")
        
        shell, grants = self.fiat_shell_grants_log.pop()
        
        # Remove from visible log too (keep both in sync)
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
        """
        Rollback n kernel steps using inverse stepping.
        
        Args:
            n: number of steps to rollback
        """
        from src.router.constants import GENE_MIC_S
        
        n = min(n, self.kernel.step)
        for _ in range(n):
            if not self.byte_log:
                break
            b = self.byte_log.pop()
            self.kernel.step_byte_inverse(b)
        self.kernel.last_byte = self.byte_log[-1] if self.byte_log else GENE_MIC_S
        assert self.kernel.step == len(self.byte_log)

    def sign_bundle(self, bundle_json_bytes: bytes, private_key: Any) -> bytes:
        """
        Sign bundle JSON bytes using Ed25519.
        
        Args:
            bundle_json_bytes: JSON bytes of bundle (without signature fields)
            private_key: Ed25519PrivateKey instance
            
        Returns:
            Signature bytes
        """
        if Ed25519PrivateKey is None:
            raise ImportError("cryptography package required for bundle signing")
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"private_key must be Ed25519PrivateKey, got {type(private_key)}")
        
        digest = hashlib.sha256(bundle_json_bytes).digest()
        return private_key.sign(digest)

    def verify_bundle_signature(self, bundle_json_bytes: bytes, signature: bytes, public_key: Any) -> bool:
        """
        Verify bundle JSON bytes signature using Ed25519.
        
        Args:
            bundle_json_bytes: JSON bytes of bundle (without signature fields)
            signature: Signature bytes
            public_key: Ed25519PublicKey instance
            
        Returns:
            True if signature is valid, False otherwise
        """
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
        
        # Reset fiat substrate
        self.sealer = RouterKernel(self.atlas_dir)
        self.fiat_grants_current.clear()
        self.fiat_shell_log.clear()
        self.fiat_shell_grants_log.clear()
        self.fiat_archive_totals.clear()
        self.fiat_identity_id_to_identity.clear()
        self.fiat_used_total = 0
        self.fiat_capacity_total = 0

    def derive_domain_counts(self) -> Dict[str, int]:
        """
        Derive domain_counts from the event log.
        
        Per GGG hierarchy:
        - Economy = Kernel (structural substrate)
        - Employment = Gyroscope (active work/principles)
        - Education = THM (measurements/displacements)
        
        Returns dict with keys: "economy", "employment", "education"
        """
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

