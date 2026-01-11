# tests/test_ecology_capacity_ledger.py
"""
Ecology Capacity Ledger Experiments

These tests explore how to:
- realise atomic abundance as a concrete MU capacity envelope
- define replayable "capacity windows" anchored by kernel states
- allocate MU to identity-anchored genealogies inside that envelope
- maintain a simple global ecology capacity ledger that is replayable

They DO NOT change kernel physics or core app modules.
They are exploratory design tests in the spirit of test_substrate.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import numpy as np
import pytest

from src.router.kernel import RouterKernel

ATLAS_DIR = Path(__file__).parent.parent / "data" / "atlas"


# ---------------------------------------------------------------------------
# 0. Physical and economic constants (copied from existing reports/tests)
# ---------------------------------------------------------------------------

# Atomic + kernel throughput as in test_moments.py (F_total ≈ 22.06e15 / sec)
F_TOTAL_PER_SEC = 22_062_316_248_000_000  # structural micro-state references per second

SECONDS_PER_YEAR = 365 * 24 * 60 * 60  # 31,536,000
WORLD_POP = 8_100_000_000
UHI_PER_YEAR_MU = 87_600  # from Moments spec: 4 hours/day at 60 MU/hour

# Conservative mapping: 1 structural micro-state reference == 1 MU
# (this is intentionally conservative; real mapping could be much looser)
CAPACITY_PER_YEAR_MU = F_TOTAL_PER_SEC * SECONDS_PER_YEAR


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def route_bytes(payload: bytes) -> str:
    """Route a payload through a fresh kernel and return state hex."""
    k = RouterKernel(ATLAS_DIR)
    k.step_payload(payload)
    return k.signature().state_hex


def identity_commitment(name: str) -> str:
    """
    Compute a stable identity anchor:
    - hash the name
    - route the hash through the kernel
    - use the resulting state_hex as the identity commitment
    """
    seed = f"identity:{name}".encode()
    h = hashlib.sha256(seed).digest()
    return route_bytes(h)


@dataclass(frozen=True)
class AllocationEvent:
    """
    A single allocation of MU to an identity within a capacity window.
    """
    identity: str           # human-readable id
    identity_commit: str    # 3-byte state_hex commitment
    mu_allocated: int       # MU allocated in this window


@dataclass(frozen=True)
class CapacityWindow:
    """
    A capacity window in the ecology substrate.

    header: arbitrary contextual header (e.g. "year:2026", "epoch:1")
    window_state: kernel state_hex after routing header || allocation receipts
    total_capacity_MU: MU capacity available in this window (from physics)
    used_capacity_MU: sum of MU allocations in this window
    free_capacity_MU: remaining MU capacity
    """
    header: bytes
    window_state: str
    total_capacity_MU: int
    used_capacity_MU: int
    free_capacity_MU: int


def build_capacity_window(
    header: bytes,
    alloc_events: List[AllocationEvent],
    total_capacity_MU: int,
) -> CapacityWindow:
    """
    Construct a capacity window:
    - encode header + receipts into a payload
    - route through kernel to obtain a window_state commitment
    - compute used and free capacity
    """
    # Encode receipts deterministically: sort by identity_commit for canonical ordering
    receipts = []
    for ev in sorted(alloc_events, key=lambda e: e.identity_commit):
        # Each receipt is (identity_commit || mu_allocated as 8 bytes big-endian)
        mu_bytes = ev.mu_allocated.to_bytes(8, "big", signed=False)
        receipts.append(bytes.fromhex(ev.identity_commit) + mu_bytes)
    payload = header + b"".join(receipts)

    window_state = route_bytes(payload)
    used = sum(ev.mu_allocated for ev in alloc_events)
    free = total_capacity_MU - used

    return CapacityWindow(
        header=header,
        window_state=window_state,
        total_capacity_MU=total_capacity_MU,
        used_capacity_MU=used,
        free_capacity_MU=free,
    )


def replay_capacity_window(
    header: bytes,
    alloc_events: List[AllocationEvent],
    total_capacity_MU: int,
) -> CapacityWindow:
    """
    Replay function: same as build_capacity_window, provided to emphasise
    determinism in tests.
    """
    return build_capacity_window(header, alloc_events, total_capacity_MU)


# ---------------------------------------------------------------------------
# 1. Test: Physical capacity vs global UHI for one year
# ---------------------------------------------------------------------------

def test_01_global_uhi_vs_physical_capacity_one_year():
    """
    Check that one year of UHI for the full world population fits comfortably
    inside the conservative capacity mapping.

    This is a one-year analogue of the Millennium feasibility test.
    """
    total_uhi_year = WORLD_POP * UHI_PER_YEAR_MU
    usage_fraction = total_uhi_year / CAPACITY_PER_YEAR_MU

    print("\n=== GLOBAL UHI VS PHYSICAL CAPACITY (one year) ===")
    print(f"F_total_per_sec      : {F_TOTAL_PER_SEC:>20,d}")
    print(f"Seconds/year         : {SECONDS_PER_YEAR:>20,d}")
    print(f"Capacity/year (MU)   : {CAPACITY_PER_YEAR_MU:>20,d}")
    print(f"Total UHI/year (MU)  : {total_uhi_year:>20,d}")
    print(f"Usage fraction       : {usage_fraction:.12e}")

    # We expect this to be extremely small; assert < 1e-6 conservatively
    assert usage_fraction < 1e-6


# ---------------------------------------------------------------------------
# 2. Test: One ecology capacity window with identity-anchored allocations
# ---------------------------------------------------------------------------

def test_02_single_capacity_window_with_identity_allocations():
    """
    Build a single capacity window (e.g. a year) with a handful of identities.
    Allocate MU (e.g. UHI + tier increments) and ensure:
    - sum of allocations <= physical capacity
    - window_state is deterministically reproducible
    - identity commitments are distinct and stable
    """
    print("\n=== SINGLE ECOLOGY CAPACITY WINDOW ===")

    header = b"ecology:year:2026"
    total_capacity_MU = CAPACITY_PER_YEAR_MU  # could be scaled down in practice

    # Define a small set of identities and allocations (UHI + tiers)
    identities = ["alice", "bob", "carol", "dave"]
    tiers = {
        "alice": UHI_PER_YEAR_MU * 3,   # e.g. Tier 3
        "bob":   UHI_PER_YEAR_MU * 2,   # Tier 2
        "carol": UHI_PER_YEAR_MU * 1,   # Tier 1
        "dave":  UHI_PER_YEAR_MU * 1,   # Tier 1
    }

    events: List[AllocationEvent] = []
    for name in identities:
        commit = identity_commitment(name)
        mu = tiers[name]
        events.append(
            AllocationEvent(
                identity=name,
                identity_commit=commit,
                mu_allocated=mu,
            )
        )

    # Build window
    win = build_capacity_window(header, events, total_capacity_MU)

    print(f"Header              : {header}")
    print(f"Window state        : {win.window_state}")
    print(f"Total capacity (MU) : {win.total_capacity_MU}")
    print(f"Used capacity (MU)  : {win.used_capacity_MU}")
    print(f"Free capacity (MU)  : {win.free_capacity_MU}")

    # Check: all identity commitments distinct
    commits = [e.identity_commit for e in events]
    assert len(set(commits)) == len(commits)

    # Check: allocations fit within capacity
    assert win.used_capacity_MU <= win.total_capacity_MU

    # Check: replay is deterministic
    win2 = replay_capacity_window(header, events, total_capacity_MU)
    assert win2.window_state == win.window_state
    assert win2.used_capacity_MU == win.used_capacity_MU
    assert win2.free_capacity_MU == win.free_capacity_MU

    print("✓ single capacity window is deterministic and within physical envelope")


# ---------------------------------------------------------------------------
# 3. Test: Global ecology capacity ledger over multiple windows
# ---------------------------------------------------------------------------

@dataclass
class EcologyCapacityLedger:
    """
    Minimal ecology capacity ledger:
    - records per-identity MU allocations over multiple windows
    - tracks used vs total capacity over all windows
    - is replayable from (header, alloc_events) logs
    """
    per_identity_MU: Dict[str, int]
    total_capacity_MU: int
    used_capacity_MU: int
    free_capacity_MU: int


def build_ecology_ledger(
    windows: List[Tuple[bytes, List[AllocationEvent]]],
    capacity_per_window_MU: int,
) -> EcologyCapacityLedger:
    """
    Aggregate a set of capacity windows into a global ledger.
    """
    per_identity: Dict[str, int] = {}
    used_total = 0
    total_capacity = 0

    for header, events in windows:
        win = build_capacity_window(header, events, capacity_per_window_MU)
        total_capacity += win.total_capacity_MU
        used_total += win.used_capacity_MU
        for ev in events:
            per_identity[ev.identity] = per_identity.get(ev.identity, 0) + ev.mu_allocated

    free_total = total_capacity - used_total
    return EcologyCapacityLedger(
        per_identity_MU=per_identity,
        total_capacity_MU=total_capacity,
        used_capacity_MU=used_total,
        free_capacity_MU=free_total,
    )


def test_03_global_ecology_capacity_ledger_replay():
    """
    Build a small sequence of capacity windows (e.g. 3 years),
    each with a simple allocation pattern. Verify that:
    - the global ledger is within the cumulative physical envelope
    - replay from the same (header, events) pairs reproduces the same ledger
    """
    print("\n=== GLOBAL ECOLOGY CAPACITY LEDGER OVER MULTIPLE WINDOWS ===")

    capacity_per_window_MU = CAPACITY_PER_YEAR_MU  # one-year windows
    headers = [b"ecology:year:2026", b"ecology:year:2027", b"ecology:year:2028"]

    identities = ["alice", "bob", "carol"]
    # Simple fixed pattern: alice Tier 3, bob Tier 2, carol Tier 1 each year
    tiers = {
        "alice": UHI_PER_YEAR_MU * 3,
        "bob":   UHI_PER_YEAR_MU * 2,
        "carol": UHI_PER_YEAR_MU * 1,
    }

    windows: List[Tuple[bytes, List[AllocationEvent]]] = []
    for hdr in headers:
        evs: List[AllocationEvent] = []
        for name in identities:
            commit = identity_commitment(name)
            mu = tiers[name]
            evs.append(
                AllocationEvent(
                    identity=name,
                    identity_commit=commit,
                    mu_allocated=mu,
                )
            )
        windows.append((hdr, evs))

    ledger = build_ecology_ledger(windows, capacity_per_window_MU)

    print(f"Total capacity (MU) : {ledger.total_capacity_MU}")
    print(f"Used capacity (MU)  : {ledger.used_capacity_MU}")
    print(f"Free capacity (MU)  : {ledger.free_capacity_MU}")
    print(f"Per-identity MU     : {ledger.per_identity_MU}")

    # Check: allocations fit within capacity
    assert ledger.used_capacity_MU <= ledger.total_capacity_MU

    # Expected per-identity totals across 3 windows
    assert ledger.per_identity_MU["alice"] == UHI_PER_YEAR_MU * 3 * len(headers)
    assert ledger.per_identity_MU["bob"]   == UHI_PER_YEAR_MU * 2 * len(headers)
    assert ledger.per_identity_MU["carol"] == UHI_PER_YEAR_MU * 1 * len(headers)

    # Replay: rebuild from same windows and compare
    ledger2 = build_ecology_ledger(windows, capacity_per_window_MU)
    assert ledger2.total_capacity_MU == ledger.total_capacity_MU
    assert ledger2.used_capacity_MU == ledger.used_capacity_MU
    assert ledger2.free_capacity_MU == ledger.free_capacity_MU
    assert ledger2.per_identity_MU == ledger.per_identity_MU

    print("✓ global ecology capacity ledger is deterministic and within envelope")


# ---------------------------------------------------------------------------
# 4. Test: Tampering with allocations is visible under replay
# ---------------------------------------------------------------------------

def test_04_tampering_in_capacity_window_is_detectable():
    """
    Show that if someone tampers with an allocation amount for an identity
    in a logged window, replay will detect it via:
    - change in used_capacity_MU
    - change in per_identity_MU totals
    - (optionally) different window_state, if header+receipts are re-encoded
    """
    print("\n=== TAMPERING IN CAPACITY WINDOW ===")

    header = b"ecology:year:2026"
    capacity_MU = CAPACITY_PER_YEAR_MU

    names = ["alice", "bob"]
    tiers = {
        "alice": UHI_PER_YEAR_MU * 2,
        "bob":   UHI_PER_YEAR_MU * 1,
    }

    evs: List[AllocationEvent] = []
    for name in names:
        commit = identity_commitment(name)
        evs.append(
            AllocationEvent(
                identity=name,
                identity_commit=commit,
                mu_allocated=tiers[name],
            )
        )

    # Original window and ledger
    win_orig = build_capacity_window(header, evs, capacity_MU)
    ledger_orig = build_ecology_ledger([(header, evs)], capacity_MU)

    # Tamper: change alice's allocation without changing header/identity names
    evs_tampered = evs.copy()
    evs_tampered[0] = AllocationEvent(
        identity="alice",
        identity_commit=evs[0].identity_commit,
        mu_allocated=tiers["alice"] * 10,  # inflated
    )

    win_tampered = build_capacity_window(header, evs_tampered, capacity_MU)
    ledger_tampered = build_ecology_ledger([(header, evs_tampered)], capacity_MU)

    print(f"Original used (MU)   : {win_orig.used_capacity_MU}")
    print(f"Tampered used (MU)   : {win_tampered.used_capacity_MU}")
    print(f"Original per-id MU   : {ledger_orig.per_identity_MU}")
    print(f"Tampered per-id MU   : {ledger_tampered.per_identity_MU}")
    print(f"Original window state: {win_orig.window_state}")
    print(f"Tampered window state: {win_tampered.window_state}")

    # Detect tampering:
    assert win_tampered.used_capacity_MU != win_orig.used_capacity_MU
    assert ledger_tampered.per_identity_MU["alice"] != ledger_orig.per_identity_MU["alice"]
    # Encoding difference usually changes window_state as well
    assert win_tampered.window_state != win_orig.window_state

    print("✓ tampering in capacity window is visible under replay")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", "-v", __file__]))