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

4) Fiat Capacity (abundance demonstration):
   - Demonstrates a conservative “capacity upper bound” using:
       F_total = atomic_hz × kernel_steps_per_sec_avg
   - Shows that, under a very conservative mapping 1 micro-state == 1 MU,
     funding UHI for the whole world for 1,000 years uses a vanishing fraction
     of available capacity.

This file is deliberately:
- deterministic
- implementation-light (no need to load atlas files)
- oriented to explaining the system as a physics-backed, post-scarcity economy

Run:
  python -m pytest -v -s tests/test_moments.py
"""

from __future__ import annotations

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
ROUTER_ONTLOGY_SIZE = 65_536
ROUTER_BYTE_ALPHABET = 256

# CGM / Router aperture constants
A_KERNEL = 5 / 256  # exact discrete aperture shadow from kernel structure
# Use a high-precision CGM target from earlier CGM/physics reporting (approx).
A_STAR = 0.020699553813

# Atomic second anchor (SI)
ATOMIC_HZ_CS133 = 9_192_631_770  # Hz

# Representative Router throughput for capacity demonstrations (not a unit definition)
KERNEL_STEPS_PER_SEC_AVG = 2_400_000

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


def fiat_capacity_microstates_per_sec(
    atomic_hz: int = ATOMIC_HZ_CS133,
    kernel_steps_per_sec: int = KERNEL_STEPS_PER_SEC_AVG,
) -> int:
    return int(atomic_hz) * int(kernel_steps_per_sec)


@dataclass(frozen=True)
class FundingHorizon:
    years: int
    population: int
    uhi_mu_per_person_per_year: int
    available_units_per_year: int  # in "microstates/year" or any chosen unit
    mapping_note: str

    @property
    def needed_units_per_year(self) -> int:
        return int(self.population) * int(self.uhi_mu_per_person_per_year)

    @property
    def needed_units_over_horizon(self) -> int:
        return self.needed_units_per_year * int(self.years)

    @property
    def available_units_over_horizon(self) -> int:
        return int(self.available_units_per_year) * int(self.years)

    @property
    def used_fraction(self) -> float:
        return self.needed_units_over_horizon / self.available_units_over_horizon

    @property
    def used_percent(self) -> float:
        return 100.0 * self.used_fraction

    @property
    def surplus_units_over_horizon(self) -> int:
        return self.available_units_over_horizon - self.needed_units_over_horizon


# ----------------------------
# Tests: Physics/Kernel Anchors
# ----------------------------

def test_router_static_structure_anchors():
    """
    The Router's canonical discrete structure used by the economy narrative.
    These are structural constants of the Router specification.
    """
    assert ROUTER_ONTLOGY_SIZE == 65_536
    assert ROUTER_BYTE_ALPHABET == 256

    print("\n----------")
    print("Router Anchors")
    print("----------")
    print(f"Ontology size |Ω|: {format_int(ROUTER_ONTLOGY_SIZE)}")
    print(f"Byte alphabet: {format_int(ROUTER_BYTE_ALPHABET)}")


def test_aperture_shadow_a_kernel_close_to_a_star():
    """
    The Router has an intrinsic discrete small-openness constant:
      A_kernel = 5/256

    CGM target:
      A* ≈ 0.020699553813

    We only assert “close” at the order-of-magnitude level to avoid
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
# Tests: Fiat Capacity and Abundance Demonstrations
# ----------------------------

def test_fiat_capacity_upper_bound_from_atomic_and_kernel_rates():
    """
    Fiat Capacity demonstration (upper bound):

      F_total = atomic_hz × kernel_steps_per_sec_avg

    This is used as a capacity demonstration, not as a fragile economic unit
    that changes with CPU speed.
    """
    total = fiat_capacity_microstates_per_sec()

    print("\n----------")
    print("Fiat Capacity Upper Bound")
    print("----------")
    print(f"Atomic Hz:          {format_int(ATOMIC_HZ_CS133)}")
    print(f"Kernel steps/sec:   {format_int(KERNEL_STEPS_PER_SEC_AVG)}")
    print(f"F_total (/sec):     {format_large_number(total)}")

    assert total == 22_062_316_248_000_000


def test_millennium_uhi_feasibility_under_conservative_mapping():
    """
    Millennium feasibility demonstration.

    We adopt a deliberately conservative demonstration mapping:
      1 micro-state of Fiat Capacity == 1 MU

    This is not required by the spec, but it is a useful upper-bound
    demonstration of abundance.

    We then compare:
      needed MU for UHI to world population over 1,000 years
      vs
      available micro-states over 1,000 years

    The used fraction should be extremely small.
    """
    population = 8_100_000_000
    horizon_years = 1_000

    uhi_mu_year = compute_uhi_mu_per_year()  # 87,600 MU/year
    cap_per_sec = fiat_capacity_microstates_per_sec()
    cap_per_year = cap_per_sec * SECONDS_PER_YEAR

    horizon = FundingHorizon(
        years=horizon_years,
        population=population,
        uhi_mu_per_person_per_year=uhi_mu_year,
        available_units_per_year=cap_per_year,
        mapping_note="Conservative demonstration mapping: 1 micro-state == 1 MU",
    )

    print("\n----------")
    print("Millennium UHI Feasibility (Conservative Mapping)")
    print("----------")
    print(horizon.mapping_note)
    print(f"Population:                      {format_int(population)}")
    print(f"UHI per person per year (MU):    {format_int(uhi_mu_year)}")
    print(f"Needed per year (MU):            {format_large_number(horizon.needed_units_per_year)}")
    print(f"Available per year (units):      {format_large_number(cap_per_year)}")
    print(f"Used % over {horizon_years} years:        {format_pct(horizon.used_percent)}")
    print(f"Surplus over {horizon_years} years (units): {format_large_number(horizon.surplus_units_over_horizon)}")

    # The old runs show ~1.02e-7% used under this mapping.
    assert horizon.used_percent < 1e-4  # < 0.0001% is a very conservative bound


def test_notional_surplus_allocation_12_divisions():
    """
    Notional surplus allocation across 12 structural divisions:
      3 domains × 4 Gyroscope capacities

    This is a planning representation: it does not imply stored tokens.
    It simply partitions headroom available under the conservative mapping.
    """
    population = 8_100_000_000
    horizon_years = 1_000

    uhi_mu_year = compute_uhi_mu_per_year()
    cap_per_year = fiat_capacity_microstates_per_sec() * SECONDS_PER_YEAR

    horizon = FundingHorizon(
        years=horizon_years,
        population=population,
        uhi_mu_per_person_per_year=uhi_mu_year,
        available_units_per_year=cap_per_year,
        mapping_note="Conservative demonstration mapping: 1 micro-state == 1 MU",
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
    print(f"Surplus (units):   {format_large_number(surplus)}")
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