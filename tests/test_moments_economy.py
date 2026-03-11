# tests/test_moments_economy.py
"""
Moments Economy: economic definitions, capacity, and medium integrity.

Tests the economic layer of the Moments Economy Architecture:
- MU (Moment-Unit) definition and base-60 rate
- UHI (Unconditional High Income) amounts
- Tier multipliers and mnemonics
- CSM (Common Source Moment) capacity with |Omega| = 4096
- Coverage, resilience, and tier distribution analysis
- Identity Anchors, Grants, Shells, Archives, and replay verification

Physical constants and Router structure proofs live in test_moments_physics.py
and test_moments_physics_2.py. This file tests economic definitions,
capacity demonstrations, and medium-layer correctness.

All tests are deterministic, require no atlas files, and depend only on
src.constants and src.kernel.

Run:
    python -m pytest tests/test_moments_economy.py -v -s
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

PROGRAM_ROOT = Path(__file__).parent.parent
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from src.kernel import Gyroscopic
from tests._moments_utils import Grant, Shell, identity_anchor, _make_shell


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(num: float) -> str:
    """Human-readable large number formatting."""
    if num >= 1e18:
        return f"{num / 1e18:,.2f} quintillion ({num:,.0f})"
    if num >= 1e15:
        return f"{num / 1e15:,.2f} quadrillion ({num:,.0f})"
    if num >= 1e12:
        return f"{num / 1e12:,.2f} trillion ({num:,.0f})"
    if num >= 1e9:
        return f"{num / 1e9:,.2f} billion ({num:,.0f})"
    if num >= 1e6:
        return f"{num / 1e6:,.2f} million ({num:,.0f})"
    return f"{num:,.2f}"


# ---------------------------------------------------------------------------
# Economic constants (spec-aligned, no kernel dependency)
# ---------------------------------------------------------------------------

MU_PER_MINUTE: int = 1
MU_PER_HOUR: int = 60
HOURS_PER_DAY: int = 24
DAYS_PER_YEAR: int = 365
SECONDS_PER_HOUR: int = 3600

UHI_HOURS_PER_DAY: int = 4
UHI_PER_DAY: int = UHI_HOURS_PER_DAY * MU_PER_HOUR
UHI_PER_YEAR: int = UHI_PER_DAY * DAYS_PER_YEAR

TIER_MULTIPLIERS: dict[str, int] = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3,
    "Tier 4": 60,
}

POPULATION: int = 8_100_000_000

# ---------------------------------------------------------------------------
# Physical and Router constants
# ---------------------------------------------------------------------------

F_CS: int = 9_192_631_770
OMEGA_SIZE: int = 4096
HORIZON_SIZE: int = 64


def n_phys() -> float:
    """N_phys = (4/3) pi f_Cs^3. Speed of light cancels."""
    return (4.0 / 3.0) * math.pi * (F_CS ** 3)


def csm_total() -> float:
    """Common Source Moment = N_phys / |Omega|. Fixed total capacity."""
    return n_phys() / float(OMEGA_SIZE)


# ---------------------------------------------------------------------------
# Coverage dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coverage:
    """Coverage analysis for a given capacity and demand."""
    total_mu: float
    annual_demand_mu: float

    @property
    def years(self) -> float:
        return self.total_mu / self.annual_demand_mu

    @property
    def usage_pct(self) -> float:
        return 100.0 * self.annual_demand_mu / self.total_mu


# ---------------------------------------------------------------------------
# Medium-layer data structures (Grant, Shell from _moments_utils)
# ---------------------------------------------------------------------------

@dataclass
class Archive:
    """Long-horizon aggregation of Shells."""
    shells: list[Shell] = field(default_factory=list)

    @property
    def per_identity_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for shell in self.shells:
            for g in shell.grants:
                totals[g.identity_label] = totals.get(g.identity_label, 0) + g.amount_mu
        return totals

    @property
    def total_used(self) -> int:
        return sum(s.used_capacity_mu for s in self.shells)


# ===================================================================
# Part 1: Economic definitions
# ===================================================================

def test_router_capacity_anchors():
    """Structural constants used by the economy layer."""
    assert OMEGA_SIZE == 4096
    assert HORIZON_SIZE == 64
    assert HORIZON_SIZE * HORIZON_SIZE == OMEGA_SIZE
    assert F_CS == 9_192_631_770


def test_mu_definition_and_base_rate():
    """
    MU is defined so that at the base rate:
      1 MU per minute, 60 MU per hour.
    """
    assert MU_PER_MINUTE == 1
    assert MU_PER_HOUR == 60

    print("\n--- MU Base Rate ---")
    print(f"  MU/minute: {MU_PER_MINUTE}")
    print(f"  MU/hour:   {MU_PER_HOUR}")


def test_uhi_amounts():
    """
    UHI = 4 hours/day at base rate, every day.
      Daily:  240 MU
      Annual: 87,600 MU
    """
    assert UHI_PER_DAY == 240
    assert UHI_PER_YEAR == 87_600

    print("\n--- UHI ---")
    print(f"  Daily:  {UHI_PER_DAY:,} MU")
    print(f"  Annual: {UHI_PER_YEAR:,} MU")


def test_tier_multipliers():
    """
    Tiers are multiples of UHI:
      Tier 1: 1x, Tier 2: 2x, Tier 3: 3x, Tier 4: 60x.
    """
    expected = {
        "Tier 1": 87_600,
        "Tier 2": 175_200,
        "Tier 3": 262_800,
        "Tier 4": 5_256_000,
    }
    print("\n--- Tier Schedule ---")
    for name, mult in TIER_MULTIPLIERS.items():
        annual = mult * UHI_PER_YEAR
        assert annual == expected[name]
        print(f"  {name}: {mult}x = {annual:,} MU/year")


def test_tier4_mnemonic():
    """Tier 4 = 14,400 seconds/day x 365 = 5,256,000."""
    seconds_in_4h = 4 * SECONDS_PER_HOUR
    mnemonic = seconds_in_4h * DAYS_PER_YEAR
    tier4 = TIER_MULTIPLIERS["Tier 4"] * UHI_PER_YEAR
    assert seconds_in_4h == 14_400
    assert mnemonic == 5_256_000
    assert tier4 == mnemonic


def test_work_week_is_not_tier_definition():
    """Tiers are UHI multipliers, not work schedules."""
    illustrative = 4 * 4 * 52 * MU_PER_HOUR
    assert illustrative == 49_920
    assert illustrative != UHI_PER_YEAR


# ===================================================================
# Part 2: CSM capacity and abundance
# ===================================================================

def test_csm_capacity():
    """CSM = N_phys / 4096. Fixed total capacity."""
    csm = csm_total()
    assert csm > 7e26
    assert csm < 8e26

    print("\n--- CSM Capacity ---")
    print(f"  f_Cs:    {F_CS:,} Hz")
    print(f"  N_phys:  {n_phys():.6e}")
    print(f"  |Omega|: {OMEGA_SIZE:,}")
    print(f"  CSM:     {_fmt(csm)} MU")


def test_global_uhi_coverage():
    """Coverage ~ 1.12 trillion years."""
    csm = csm_total()
    demand = float(POPULATION) * float(UHI_PER_YEAR)
    cov = Coverage(total_mu=csm, annual_demand_mu=demand)
    assert cov.years > 1e12

    print("\n--- Global UHI Coverage ---")
    print(f"  Population:    {POPULATION:,}")
    print(f"  Annual demand: {_fmt(demand)} MU")
    print(f"  CSM total:     {_fmt(csm)} MU")
    print(f"  Coverage:      {cov.years:.2e} years")


def test_adversarial_resilience():
    """An adversary needs ~11 billion x annual UHI to consume 1% of CSM."""
    csm = csm_total()
    demand = float(POPULATION) * float(UHI_PER_YEAR)
    threshold = 0.01 * csm
    multiplier = threshold / demand
    assert multiplier > 1e7

    print("\n--- Adversarial Resilience ---")
    print(f"  1% of CSM requires: {multiplier:,.0f}x annual demand")


def test_realistic_tier_distributions():
    """Weighted demand under conservative/plausible/generous scenarios."""
    csm = csm_total()
    distributions: dict[str, dict[str, float]] = {
        "Conservative": {"Tier 1": 95.0, "Tier 2": 4.0, "Tier 3": 0.9, "Tier 4": 0.1},
        "Plausible":    {"Tier 1": 90.0, "Tier 2": 8.0, "Tier 3": 1.5, "Tier 4": 0.5},
        "Generous":     {"Tier 1": 85.0, "Tier 2": 12.0, "Tier 3": 2.5, "Tier 4": 0.5},
    }
    results: dict[str, float] = {}
    for name, dist in distributions.items():
        assert abs(sum(dist.values()) - 100.0) < 0.01
        weighted_mult = sum(
            (dist[t] / 100.0) * TIER_MULTIPLIERS[t] for t in TIER_MULTIPLIERS
        )
        demand = float(POPULATION) * UHI_PER_YEAR * weighted_mult
        cov = Coverage(total_mu=csm, annual_demand_mu=demand)
        results[name] = cov.years

    for name, years in results.items():
        assert years > 1e11
    assert results["Conservative"] > results["Plausible"] > results["Generous"]


def test_notional_surplus_allocation():
    """CSM partitioned: 3 domains x 4 capacities = 12 divisions."""
    csm = csm_total()
    reserve = float(UHI_PER_YEAR) * float(POPULATION) * 1000
    surplus = csm - reserve
    per_div = surplus / 12
    assert surplus > 0
    assert per_div > 0


# ===================================================================
# Part 3: Identity Anchors
# ===================================================================

def test_identity_anchor_determinism():
    """Same name always produces the same anchor."""
    id1, anchor1 = identity_anchor("alice")
    id2, anchor2 = identity_anchor("alice")
    assert id1 == id2
    assert anchor1 == anchor2


def test_identity_anchor_separation():
    """Different names produce different anchors."""
    _, a1 = identity_anchor("alice")
    _, a2 = identity_anchor("bob")
    assert a1 != a2


def test_identity_anchor_is_router_state():
    """Kernel anchor is a valid 6-hex-char Router state."""
    _, anchor = identity_anchor("alice")
    assert len(anchor) == 6
    int(anchor, 16)  # must parse as hex


# ===================================================================
# Part 4: Grants
# ===================================================================

def test_grant_canonical_receipt():
    """Grant produces a deterministic canonical receipt."""
    ident, anchor = identity_anchor("alice")
    g = Grant("alice", ident, anchor, UHI_PER_YEAR)
    receipt = g.canonical_receipt()

    assert isinstance(receipt, bytes)
    assert len(receipt) == 32 + 6 + 8  # SHA-256 + anchor + amount

    # Same grant, same receipt
    g2 = Grant("alice", ident, anchor, UHI_PER_YEAR)
    assert g2.canonical_receipt() == receipt


def test_grant_different_amounts_different_receipts():
    """Different amounts produce different receipts."""
    ident, anchor = identity_anchor("alice")
    g1 = Grant("alice", ident, anchor, UHI_PER_YEAR)
    g2 = Grant("alice", ident, anchor, UHI_PER_YEAR * 2)
    assert g1.canonical_receipt() != g2.canonical_receipt()


# ===================================================================
# Part 5: Shells and seals
# ===================================================================

def test_shell_seal_determinism():
    """Same grants and header produce the same seal."""
    s1 = _make_shell(b"ecology:year:2026", [
        ("alice", UHI_PER_YEAR * 3),
        ("bob", UHI_PER_YEAR * 2),
    ])
    s2 = _make_shell(b"ecology:year:2026", [
        ("alice", UHI_PER_YEAR * 3),
        ("bob", UHI_PER_YEAR * 2),
    ])
    assert s1.seal == s2.seal
    assert s1.used_capacity_mu == s2.used_capacity_mu

    print("\n--- Shell Seal Determinism ---")
    print(f"  Seal: {s1.seal}")
    print(f"  Used: {s1.used_capacity_mu:,} MU")


def test_shell_seal_tamper_evidence():
    """Changing any grant changes the seal."""
    s1 = _make_shell(b"ecology:year:2026", [
        ("alice", UHI_PER_YEAR * 3),
        ("bob", UHI_PER_YEAR * 2),
    ])
    s_tampered = _make_shell(b"ecology:year:2026", [
        ("alice", UHI_PER_YEAR * 30),  # inflated
        ("bob", UHI_PER_YEAR * 2),
    ])
    assert s_tampered.seal != s1.seal
    assert s_tampered.used_capacity_mu != s1.used_capacity_mu


def test_shell_seal_header_sensitivity():
    """Changing the header changes the seal."""
    s1 = _make_shell(b"ecology:year:2026", [("alice", UHI_PER_YEAR)])
    s2 = _make_shell(b"ecology:year:2027", [("alice", UHI_PER_YEAR)])
    assert s1.seal != s2.seal


def test_shell_grant_order_invariance():
    """Seal is invariant to the order grants are added."""
    s1 = _make_shell(b"test", [
        ("alice", UHI_PER_YEAR),
        ("bob", UHI_PER_YEAR * 2),
        ("carol", UHI_PER_YEAR * 3),
    ])
    s2 = _make_shell(b"test", [
        ("carol", UHI_PER_YEAR * 3),
        ("alice", UHI_PER_YEAR),
        ("bob", UHI_PER_YEAR * 2),
    ])
    assert s1.seal == s2.seal


def test_shell_capacity_accounting():
    """Used and free capacity are correct."""
    s = _make_shell(b"test", [
        ("alice", 100_000),
        ("bob", 200_000),
    ], capacity=1_000_000)
    assert s.used_capacity_mu == 300_000
    assert s.free_capacity_mu == 700_000


def test_shell_replay_verification():
    """
    The core verification procedure:
    1. Given published shell data (header, grants, seal).
    2. Independently reconstruct the canonical byte sequence.
    3. Route through a fresh Router from archetype.
    4. Compare: computed seal must match published seal.
    """
    # Publisher creates shell
    published = _make_shell(b"ecology:year:2026", [
        ("alice", UHI_PER_YEAR),
        ("bob", UHI_PER_YEAR * 2),
        ("carol", UHI_PER_YEAR * 3),
    ])
    published_seal = published.seal

    # Verifier: reconstruct from published data
    sorted_grants = sorted(published.grants, key=lambda g: g.identity_id)
    payload = published.header
    for g in sorted_grants:
        payload += g.canonical_receipt()

    r = Gyroscopic()
    sig = r.route_from_archetype(payload)
    verified_seal = sig.state_hex

    assert verified_seal == published_seal

    print("\n--- Shell Replay Verification ---")
    print(f"  Published seal: {published_seal}")
    print(f"  Verified seal:  {verified_seal}")
    print("  Match: True")


# ===================================================================
# Part 6: Archives
# ===================================================================

def test_archive_aggregation():
    """Archive accumulates per-identity totals across shells."""
    s1 = _make_shell(b"year:2026", [
        ("alice", UHI_PER_YEAR * 3),
        ("bob", UHI_PER_YEAR * 2),
    ])
    s2 = _make_shell(b"year:2027", [
        ("alice", UHI_PER_YEAR * 3),
        ("bob", UHI_PER_YEAR * 2),
    ])
    archive = Archive(shells=[s1, s2])

    totals = archive.per_identity_totals
    assert totals["alice"] == UHI_PER_YEAR * 3 * 2
    assert totals["bob"] == UHI_PER_YEAR * 2 * 2
    assert archive.total_used == s1.used_capacity_mu + s2.used_capacity_mu

    print("\n--- Archive Aggregation ---")
    print(f"  Per-identity: {totals}")
    print(f"  Total used: {archive.total_used:,} MU")


def test_archive_determinism():
    """Same shells produce the same archive totals."""
    def make_archive():
        s1 = _make_shell(b"y:2026", [("alice", 100), ("bob", 200)])
        s2 = _make_shell(b"y:2027", [("alice", 150), ("carol", 300)])
        return Archive(shells=[s1, s2])

    a1 = make_archive()
    a2 = make_archive()
    assert a1.per_identity_totals == a2.per_identity_totals
    assert a1.total_used == a2.total_used


# ===================================================================
# Part 7: Meta-routing
# ===================================================================

def test_meta_routing_determinism():
    """
    Multiple programme seals aggregated into a single root seal.
    Same seals -> same root.
    """
    bundles = [
        b"program:A|data:abc",
        b"program:B|data:def",
        b"program:C|data:ghi",
    ]

    def meta_root(payloads: list[bytes]) -> str:
        seals = []
        for p in payloads:
            r = Gyroscopic()
            sig = r.route_from_archetype(p)
            seals.append(bytes.fromhex(sig.state_hex))
        r = Gyroscopic()
        for s in seals:
            r.step_bytes(s)
        return r.signature().state_hex

    root1 = meta_root(bundles)
    root2 = meta_root(bundles)
    assert root1 == root2

    print("\n--- Meta-Routing ---")
    print(f"  Root: {root1}")


def test_meta_routing_tamper_localization():
    """Tampering with one bundle changes its leaf seal and the root."""
    def seal_payload(p: bytes) -> bytes:
        r = Gyroscopic()
        sig = r.route_from_archetype(p)
        return bytes.fromhex(sig.state_hex)

    def meta_root(seals: list[bytes]) -> str:
        r = Gyroscopic()
        for s in seals:
            r.step_bytes(s)
        return r.signature().state_hex

    bundles = [b"program:A", b"program:B", b"program:C"]
    seals = [seal_payload(b) for b in bundles]
    root_ok = meta_root(seals)

    tampered = [b"program:A", b"program:B:TAMPERED", b"program:C"]
    tampered_seals = [seal_payload(b) for b in tampered]
    root_bad = meta_root(tampered_seals)

    assert root_bad != root_ok
    diffs = [i for i, (a, b) in enumerate(zip(seals, tampered_seals)) if a != b]
    assert diffs == [1]

    print("\n--- Meta-Routing Tamper Localization ---")
    print(f"  Tamper at index: {diffs[0]}")


# ===================================================================
# Part 8: Kernel inverse stepping (rollback)
# ===================================================================

def test_kernel_inverse_stepping():
    """Forward then inverse returns to rest state."""
    r = Gyroscopic()
    payload = b"test payload for rollback"

    r.step_bytes(payload)
    assert r.state24 != 0xAAA555
    assert r.step == len(payload)

    r.step_bytes_inverse(payload)
    assert r.state24 == 0xAAA555

    print("\n--- Kernel Inverse Stepping ---")
    print(f"  Payload: {payload!r}")
    print("  Forward -> inverse -> rest: OK")


if __name__ == "__main__":
    os.chdir(PROGRAM_ROOT)
    raise SystemExit(pytest.main(["-s", "-v", __file__]))