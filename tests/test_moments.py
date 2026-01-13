"""
Moments Economy tests.

This test suite is aligned to the Moments Economy Architecture Specification.

It demonstrates, in a reproducible and audit-friendly way:

1) CGM / Router anchors:
   - Router ontology size (65,536) and byte action alphabet (256)
   - Discrete aperture shadow A_kernel = 5/256 and closeness to CGM A*
   - Atomic second anchor (Cs-133 hyperfine frequency)

2) Economic unit system:
   - MU (Moment-Unit) defined as 1 MU per minute at the base rate
   - Base rate: 60 MU/hour
   - UHI: 4 hours/day × 365 days at base rate => 87,600 MU/year

3) Tier model:
   - Tier 1: 1× UHI
   - Tier 2: 2× UHI
   - Tier 3: 3× UHI
   - Tier 4: 60× UHI, with an accessible mnemonic:
     Tier 4 annual = 5,256,000 MU = (4 hours/day in seconds) × 365

4) CSM Capacity (abundance demonstration):
   - Uses the kernel-native CSM formula: CSM = N_phys / |Ω|
     where N_phys = (4/3)π f_Cs³ (raw physical microcells in 1-second light-sphere)
   - This is hardware-independent and derived from physical constants + Router structure
   - Shows that funding UHI for the whole world for 1,000 years uses a vanishing fraction
     of available capacity.

This file is deliberately:
- deterministic
- implementation-light (no need to load atlas files)
- oriented to explaining the system as a physics-backed, post-scarcity economy

Run:
  python -m pytest -v -s tests/test_moments.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest


PROGRAM_ROOT = Path(__file__).parent.parent
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))


def format_large_number(num: float) -> str:
    """Format large numbers in human-readable format with commas and units."""
    def format_decimal(val: float) -> str:
        """Format decimal with commas in integer part."""
        parts = f"{val:.2f}".split(".")
        integer_part = int(float(parts[0]))
        decimal_part = parts[1] if len(parts) > 1 else "00"
        return f"{integer_part:,}.{decimal_part}"
    
    if num >= 1_000_000_000_000_000_000:
        return f"{format_decimal(num / 1_000_000_000_000_000_000)} quintillion ({num:,.0f})"
    elif num >= 1_000_000_000_000_000:
        return f"{format_decimal(num / 1_000_000_000_000_000)} quadrillion ({num:,.0f})"
    elif num >= 1_000_000_000_000:
        return f"{format_decimal(num / 1_000_000_000_000)} trillion ({num:,.0f})"
    elif num >= 1_000_000_000:
        return f"{format_decimal(num / 1_000_000_000)} billion ({num:,.0f})"
    elif num >= 1_000_000:
        return f"{format_decimal(num / 1_000_000)} million ({num:,.0f})"
    elif num >= 1_000:
        return f"{format_decimal(num / 1_000)} thousand ({num:,.0f})"
    else:
        return f"{num:,.2f}"


def format_currency(num: float) -> str:
    """Format currency in human-readable format."""
    def format_decimal(val: float) -> str:
        """Format decimal with commas in integer part."""
        parts = f"{val:.2f}".split(".")
        integer_part = int(float(parts[0]))
        decimal_part = parts[1] if len(parts) > 1 else "00"
        return f"{integer_part:,}.{decimal_part}"
    
    if num >= 1_000_000_000_000_000_000:
        return f"${format_decimal(num / 1_000_000_000_000_000_000)} quintillion (${num:,.0f})"
    elif num >= 1_000_000_000_000_000:
        return f"${format_decimal(num / 1_000_000_000_000_000)} quadrillion (${num:,.0f})"
    elif num >= 1_000_000_000_000:
        return f"${format_decimal(num / 1_000_000_000_000)} trillion (${num:,.0f})"
    elif num >= 1_000_000_000:
        return f"${format_decimal(num / 1_000_000_000)} billion (${num:,.0f})"
    elif num >= 1_000_000:
        return f"${format_decimal(num / 1_000_000)} million (${num:,.0f})"
    elif num >= 1_000:
        return f"${format_decimal(num / 1_000)} thousand (${num:,.0f})"
    else:
        return f"${num:,.2f}"


# ----------------------------
# Constants (spec-aligned)
# ----------------------------

# Router structural constants (verified elsewhere in physics tests)
ROUTER_ONTOLOGY_SIZE = 65_536
ROUTER_BYTE_ALPHABET = 256

# CGM / Router aperture constants
A_KERNEL = 5 / 256  # exact discrete aperture shadow from kernel structure
# Use a high-precision CGM target from earlier CGM/physics reporting (approx).
A_STAR = 0.020699553813

# Atomic second anchor (SI)
ATOMIC_HZ_CS133 = 9_192_631_770  # Hz

# Time constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_YEAR = SECONDS_PER_DAY * DAYS_PER_YEAR

# Economy anchor (base-60)
MU_PER_MINUTE_BASE = 1
MU_PER_HOUR_BASE = MU_PER_MINUTE_BASE * MINUTES_PER_HOUR  # 60 MU/hour


# ----------------------------
# Helpers
# ----------------------------

def format_int(n: int) -> str:
    return f"{n:,}"


def format_float(n: float, digits: int = 6) -> str:
    return f"{n:.{digits}f}"


def format_pct(x: float, digits: int = 10) -> str:
    return f"{x:.{digits}f}%"


def compute_uhi_mu_per_year(
    mu_per_hour: int = MU_PER_HOUR_BASE,
    uhi_hours_per_day: int = 4,
    days_per_year: int = DAYS_PER_YEAR,
) -> int:
    # UHI definition: 4 hours/day, every day, at base rate.
    return int(uhi_hours_per_day * mu_per_hour * days_per_year)


def tier_income_mu(uhi_mu_year: int, multiplier: int) -> int:
    return int(multiplier * uhi_mu_year)


def csm_capacity_per_sec() -> float:
    """
    Compute CSM (Common Source Moment) capacity per second.
    
    CSM = N_phys / |Ω|
    where N_phys = (4/3)π f_Cs³
    
    This is the kernel-native, hardware-independent capacity derived from:
    - Physical container: 1-second light-sphere volume
    - Atomic wavelength: λ_Cs = c / f_Cs
    - Router structural filter: |Ω| = 65,536
    
    The uniform division by |Ω| is forced by requiring symmetry-invariance
    under the Router's byte group action (proven transitive in test_moments_2.py).
    """
    f_Cs = ATOMIC_HZ_CS133
    N_phys = (4.0 / 3.0) * math.pi * (f_Cs ** 3)
    CSM = N_phys / float(ROUTER_ONTOLOGY_SIZE)
    return CSM


@dataclass(frozen=True)
class FundingHorizon:
    years: int
    population: int
    uhi_mu_per_person_per_year: int
    available_units_per_year: float  # CSM capacity per year
    mapping_note: str

    @property
    def needed_units_per_year(self) -> int:
        return int(self.population) * int(self.uhi_mu_per_person_per_year)

    @property
    def needed_units_over_horizon(self) -> int:
        return self.needed_units_per_year * int(self.years)

    @property
    def available_units_over_horizon(self) -> float:
        return float(self.available_units_per_year) * float(self.years)

    @property
    def used_fraction(self) -> float:
        return float(self.needed_units_over_horizon) / float(self.available_units_over_horizon)

    @property
    def used_percent(self) -> float:
        return 100.0 * self.used_fraction

    @property
    def surplus_units_over_horizon(self) -> float:
        return float(self.available_units_over_horizon) - float(self.needed_units_over_horizon)


# ----------------------------
# Tests: Physics/Kernel Anchors
# ----------------------------

def test_router_static_structure_anchors():
    """
    The Router's canonical discrete structure used by the economy narrative.
    These are structural constants of the Router specification.
    """
    assert ROUTER_ONTOLOGY_SIZE == 65_536
    assert ROUTER_BYTE_ALPHABET == 256

    print("\n----------")
    print("Router Anchors")
    print("----------")
    print(f"Ontology size |Ω|: {format_int(ROUTER_ONTOLOGY_SIZE)}")
    print(f"Byte alphabet: {format_int(ROUTER_BYTE_ALPHABET)}")


def test_aperture_shadow_a_kernel_close_to_a_star():
    """
    The Router has an intrinsic discrete small-openness constant:
      A_kernel = 5/256

    CGM target:
      A* ≈ 0.020699553813

    We only assert "close" at the order-of-magnitude level to avoid
    overfitting to a particular reported A* precision.
    """
    rel_diff = abs(A_KERNEL - A_STAR) / A_STAR

    print("\n----------")
    print("Aperture Shadow")
    print("----------")
    print(f"A_kernel (exact): {format_float(A_KERNEL, 12)} (5/256)")
    print(f"A* (CGM):         {format_float(A_STAR, 12)}")
    print(f"Relative diff:    {format_float(rel_diff * 100.0, 6)}%")

    # The physics tests show ~5.6% relative difference.
    assert rel_diff < 0.10  # within 10% is the conservative bound


def test_atomic_second_anchor_constant():
    """
    Confirms we use the SI atomic second anchor for time.
    """
    assert ATOMIC_HZ_CS133 == 9_192_631_770

    print("\n----------")
    print("Atomic Second Anchor")
    print("----------")
    print(f"Cs-133 hyperfine frequency: {format_int(ATOMIC_HZ_CS133)} Hz")


# ----------------------------
# Tests: MU, Base Rate, UHI
# ----------------------------

def test_mu_definition_and_base_rate_base60():
    """
    MU is defined so that at the base rate:
      1 MU per minute
      60 MU per hour

    This is the base-60 anchor: easy to remember and compute.
    """
    mu_per_min = MU_PER_MINUTE_BASE
    mu_per_hour = MU_PER_HOUR_BASE

    print("\n----------")
    print("Base-60 MU Anchor")
    print("----------")
    print(f"Seconds/minute: {SECONDS_PER_MINUTE}")
    print(f"Minutes/hour:   {MINUTES_PER_HOUR}")
    print(f"MU/minute:      {mu_per_min}")
    print(f"MU/hour:        {mu_per_hour}")

    assert mu_per_min == 1
    assert mu_per_hour == 60


def test_uhi_amounts_daily_and_annual():
    """
    UHI is defined as:
      4 hours/day, every day, at the base rate.

    Daily:
      4 hours × 60 MU/hour = 240 MU/day

    Annual:
      240 × 365 = 87,600 MU/year
    """
    uhi_hours_per_day = 4
    uhi_mu_day = uhi_hours_per_day * MU_PER_HOUR_BASE
    uhi_mu_year = compute_uhi_mu_per_year(uhi_hours_per_day=uhi_hours_per_day)

    print("\n----------")
    print("UHI")
    print("----------")
    print(f"UHI hours/day:   {uhi_hours_per_day}")
    print(f"UHI MU/day:      {format_int(uhi_mu_day)}")
    print(f"UHI MU/year:     {format_int(uhi_mu_year)}")

    assert uhi_mu_day == 240
    assert uhi_mu_year == 87_600


# ----------------------------
# Tests: Tiers and Mnemonics
# ----------------------------

def test_tier_multipliers_from_uhi():
    """
    Tier incomes are defined as simple multipliers of UHI.

    Tier 1: 1× UHI
    Tier 2: 2× UHI
    Tier 3: 3× UHI
    Tier 4: 60× UHI
    """
    uhi = compute_uhi_mu_per_year()
    tiers: Dict[str, int] = {
        "tier_1": 1,
        "tier_2": 2,
        "tier_3": 3,
        "tier_4": 60,
    }
    incomes = {k: tier_income_mu(uhi, m) for k, m in tiers.items()}

    print("\n----------")
    print("Tier Multipliers")
    print("----------")
    print(f"UHI (MU/year): {format_int(uhi)}")
    for tier, mult in tiers.items():
        print(f"{tier}: {mult}× -> {format_int(incomes[tier])} MU/year")

    assert incomes["tier_1"] == 87_600
    assert incomes["tier_2"] == 175_200
    assert incomes["tier_3"] == 262_800
    assert incomes["tier_4"] == 5_256_000


def test_tier4_accessible_mnemonic_one_per_second_for_four_hours_day():
    """
    Tier 4 is 60× UHI = 5,256,000 MU/year.

    A useful public mnemonic:
      4 hours/day = 14,400 seconds/day
      14,400 seconds/day × 365 days = 5,256,000

    This is an *accessible numeric reference*, not a requirement that MU == $1/sec.
    """
    tier4 = tier_income_mu(compute_uhi_mu_per_year(), 60)

    seconds_in_4_hours = 4 * SECONDS_PER_HOUR
    mnemonic_annual = seconds_in_4_hours * DAYS_PER_YEAR

    print("\n----------")
    print("Tier 4 Mnemonic")
    print("----------")
    print(f"Tier 4 (MU/year):        {format_int(tier4)}")
    print(f"4 hours/day in seconds:  {format_int(seconds_in_4_hours)}")
    print(f"Mnemonic annual seconds: {format_int(mnemonic_annual)}")

    assert seconds_in_4_hours == 14_400
    assert mnemonic_annual == 5_256_000
    assert tier4 == mnemonic_annual


def test_illustrative_work_week_is_not_the_definition_of_tiers():
    """
    This test prevents a common confusion:

    - Tier totals are defined by multipliers of UHI.
    - Work-week schedules (e.g. 4 hours/day, 4 days/week) are illustrative cultural norms,
      not the arithmetic definition of Tier 2 or Tier 3.

    Under the base rate (60 MU/hour):
      4h/day × 4d/week × 52 = 49,920 MU/year

    This does NOT equal the Tier 2 increment (which is defined as +1×UHI = +87,600).
    """
    uhi = compute_uhi_mu_per_year()
    tier2_total = tier_income_mu(uhi, 2)
    tier2_increment = tier2_total - uhi

    hours_per_day = 4
    days_per_week = 4
    weeks_per_year = 52
    illustrative_work_mu_year = hours_per_day * days_per_week * weeks_per_year * MU_PER_HOUR_BASE

    print("\n----------")
    print("Tier Definition vs Illustrative Work Week")
    print("----------")
    print(f"Tier 2 increment (MU/year):         {format_int(tier2_increment)}")
    print(f"Illustrative 4×4 work MU/year:      {format_int(illustrative_work_mu_year)}")
    print("Note: Tiers are defined by UHI multipliers, not by this schedule.")

    assert illustrative_work_mu_year == 49_920
    assert tier2_increment == 87_600
    assert illustrative_work_mu_year != tier2_increment


# ----------------------------
# Tests: CSM Capacity and Abundance Demonstrations
# ----------------------------

def test_csm_capacity_derivation():
    """
    CSM Capacity derivation (kernel-native, hardware-independent):

      N_phys = (4/3)π f_Cs³  (raw physical microcells in 1-second light-sphere)
      CSM = N_phys / |Ω|     (uniform coarse-grain per Router state)

    The uniform division by |Ω| is forced by requiring symmetry-invariance
    under the Router's byte group action (proven transitive in test_moments_2.py).

    This is hardware-independent: depends only on SI second and Router structure.
    """
    f_Cs = ATOMIC_HZ_CS133
    N_phys = (4.0 / 3.0) * math.pi * (f_Cs ** 3)
    CSM_per_sec = csm_capacity_per_sec()

    print("\n----------")
    print("CSM Capacity Derivation")
    print("----------")
    print(f"Atomic Hz (f_Cs):     {format_int(ATOMIC_HZ_CS133)}")
    print(f"N_phys = (4/3)π f_Cs³: {N_phys:.6e}")
    print(f"Router bulk |Ω|:      {format_int(ROUTER_ONTOLOGY_SIZE)}")
    print(f"CSM per second:       {CSM_per_sec:.6e} MU/sec")

    assert CSM_per_sec > 1e24
    assert CSM_per_sec < 1e26


def test_millennium_uhi_feasibility_under_csm():
    """
    Millennium feasibility demonstration using CSM capacity.

    We use the kernel-native CSM formula:
      CSM = N_phys / |Ω| = (4/3)π f_Cs³ / 65,536

    This is hardware-independent and derived from physical constants + Router structure.

    We then compare:
      needed MU for UHI to world population over 1,000 years
      vs
      available CSM capacity over 1,000 years

    The used fraction should be extremely small.
    """
    population = 8_100_000_000
    horizon_years = 1_000

    uhi_mu_year = compute_uhi_mu_per_year()  # 87,600 MU/year
    CSM_per_sec = csm_capacity_per_sec()
    CSM_per_year = CSM_per_sec * SECONDS_PER_YEAR

    horizon = FundingHorizon(
        years=horizon_years,
        population=population,
        uhi_mu_per_person_per_year=uhi_mu_year,
        available_units_per_year=CSM_per_year,
        mapping_note="CSM = N_phys / |Ω| (kernel-native, hardware-independent)",
    )

    print("\n----------")
    print("Millennium UHI Feasibility (CSM Capacity)")
    print("----------")
    print(horizon.mapping_note)
    print(f"Population:                      {format_int(population)}")
    print(f"UHI per person per year (MU):    {format_int(uhi_mu_year)}")
    print(f"Needed per year (MU):            {format_large_number(horizon.needed_units_per_year)}")
    print(f"CSM available per year (MU):     {format_large_number(CSM_per_year)}")
    print(f"Used % over {horizon_years} years:        {format_pct(horizon.used_percent)}")
    print(f"Surplus over {horizon_years} years (MU):  {format_large_number(horizon.surplus_units_over_horizon)}")

    # CSM provides massive abundance (should be < 1e-15%)
    assert horizon.used_percent < 1e-10  # < 0.0000000001% is a very conservative bound


def test_resilience_margin_and_adversarial_threshold():
    """
    Resilience margin demonstration using CSM capacity:
    
    R = (Available capacity − Legitimate demand) / Available capacity
    
    For the Moments Economy over any plausible horizon, R ≈ 0.9999999.
    
    This means the system can absorb orders-of-magnitude increases in demand—
    including adversarial demand—without approaching capacity limits.
    
    Adversarial threshold: The fraction of capacity that would need to be
    fraudulently claimed to cause any meaningful constraint.
    
    An adversary would need to successfully issue approximately 10 million times
    the entire global population's UHI for 1,000 years to consume just 1% of
    annual capacity. This is operationally impossible: there are not enough
    identities, not enough compute to generate them, and no registry would accept
    them.
    """
    population = 8_100_000_000
    horizon_years = 1_000
    
    uhi_mu_year = compute_uhi_mu_per_year()
    CSM_per_sec = csm_capacity_per_sec()
    CSM_per_year = CSM_per_sec * SECONDS_PER_YEAR
    
    horizon = FundingHorizon(
        years=horizon_years,
        population=population,
        uhi_mu_per_person_per_year=uhi_mu_year,
        available_units_per_year=CSM_per_year,
        mapping_note="CSM = N_phys / |Ω| (kernel-native, hardware-independent)",
    )
    
    # Resilience margin: R = (Available - Needed) / Available
    legitimate_demand_per_year = horizon.needed_units_per_year
    available_capacity_per_year = CSM_per_year
    
    resilience_margin = (available_capacity_per_year - legitimate_demand_per_year) / available_capacity_per_year
    
    # Adversarial threshold: What multiple of legitimate demand consumes 1% of annual capacity?
    target_usage_fraction = 0.01  # 1%
    adversarial_multiplier = (target_usage_fraction * available_capacity_per_year) / legitimate_demand_per_year
    
    # Adversarial demand over the horizon to consume 1% of annual capacity
    adversarial_demand_per_year = target_usage_fraction * available_capacity_per_year
    
    print("\n----------")
    print("Resilience Margin and Adversarial Threshold (CSM)")
    print("----------")
    print(f"Available capacity per year (CSM): {format_large_number(available_capacity_per_year)}")
    print(f"Legitimate demand per year:        {format_large_number(legitimate_demand_per_year)}")
    print(f"Resilience margin R:               {format_float(resilience_margin, 10)}")
    print(f"Resilience margin %:               {format_pct(resilience_margin * 100.0, 10)}")
    print(f"\nAdversarial threshold (1% of annual capacity):")
    print(f"  Target usage:                    {format_pct(target_usage_fraction * 100.0, 2)}")
    print(f"  Adversarial demand per year:     {format_large_number(adversarial_demand_per_year)}")
    print(f"  Multiple of legitimate demand:   {format_float(adversarial_multiplier, 2)}")
    print(f"  As integer multiple:             {format_int(int(round(adversarial_multiplier)))}×")
    print(f"\nInterpretation:")
    print(f"  An adversary would need to successfully issue approximately")
    print(f"  {format_int(int(round(adversarial_multiplier / 1_000_000)))} million times the entire")
    print(f"  global population's UHI for {format_int(horizon_years)} years to consume")
    print(f"  just {format_pct(target_usage_fraction * 100.0, 0)} of annual capacity.")
    print(f"  This is operationally impossible.")
    
    # Resilience margin should be very close to 1.0 (approximately 0.9999999)
    # With CSM capacity being so massive, it may round to exactly 1.0 due to floating point precision
    assert resilience_margin > 0.999999  # At least 99.9999%
    assert resilience_margin <= 1.0  # Allow exactly 1.0 due to floating point precision
    
    # Adversarial multiplier should be on the order of 10^15 or larger with CSM
    assert adversarial_multiplier > 1_000_000_000  # At least 1 billion times


def test_notional_surplus_allocation_12_divisions():
    """
    Notional surplus allocation across 12 structural divisions:
      3 domains × 4 Gyroscope capacities

    This is a planning representation: it does not imply stored tokens.
    It simply partitions headroom available under the CSM capacity.
    """
    population = 8_100_000_000
    horizon_years = 1_000

    uhi_mu_year = compute_uhi_mu_per_year()
    CSM_per_year = csm_capacity_per_sec() * SECONDS_PER_YEAR

    horizon = FundingHorizon(
        years=horizon_years,
        population=population,
        uhi_mu_per_person_per_year=uhi_mu_year,
        available_units_per_year=CSM_per_year,
        mapping_note="CSM = N_phys / |Ω| (kernel-native, hardware-independent)",
    )

    surplus = horizon.surplus_units_over_horizon

    domains = ["Economy", "Employment", "Education"]
    capacities = ["GM", "ICu", "IInter", "ICo"]
    divisions = [(d, c) for d in domains for c in capacities]

    per_division = surplus // len(divisions)

    print("\n----------")
    print("Notional Surplus Allocation (12 Divisions)")
    print("----------")
    print(f"Horizon years:     {horizon_years:,}")
    print(f"Divisions:         {len(divisions)} (3 domains × 4 capacities)")
    print(f"Surplus (MU):      {format_large_number(surplus)}")
    print(f"Per division:      {format_large_number(per_division)}")
    print("\nSample divisions:")
    for i in range(min(6, len(divisions))):
        d, c = divisions[i]
        print(f"  {d:12} × {c:6}: {format_large_number(per_division)}")

    assert len(divisions) == 12
    assert surplus > 0
    assert per_division > 0


if __name__ == "__main__":
    os.chdir(PROGRAM_ROOT)
    raise SystemExit(pytest.main(["-s", "-v", __file__]))
