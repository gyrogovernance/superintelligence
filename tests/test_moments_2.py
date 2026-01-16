"""
Moments Economy: Light-sphere → Router conversion lattice (focused, rigorous).

This file is ONLY about the conversion step needed by Moments Economy:

  Physical microcells in a 1-second causal container (light-sphere at Cs tick)
  must be mapped into a discrete, CGM-consistent governance phase space (Router Ω).

We certify (with real atlas + real kernel):

(1) Raw physical microcell count:
      N_phys = V_1s / λ_Cs^3 = (4/3)π f_Cs^3
    and this count is invariant under choice of c because c cancels.

(2) Router Ω as a conversion lattice:
      In mask coordinates (u,v) relative to archetype,
      Ω is exactly C × C, where C is the 256-element mask code.

(3) Strong isotropy statement INSIDE Ω:
      The distribution of d = u XOR v across Ω is EXACTLY uniform over C:
        each d ∈ C occurs exactly 256 times; all d ∉ C occur 0 times.
    (This is stronger and more relevant than "CV < 1" style tests.)

(4) Regular action of the even-word subgroup (2 bytes):
      For any start state s, the map (x,y) ↦ T_y(T_x(s)) is a bijection onto Ω.
    We certify this for several start states.
    This regularity forces uniform coarse-graining:
      ASSUMPTION: The physical microcell capacity should be distributed over Router
      states using a measure that is invariant under the Router's symmetries
      (the action of the byte group / even-word subgroup).
      Given the transitive action (proven above), uniformity is forced:
      any symmetry-invariant measure on Ω must be uniform.
      Therefore: per-state share = N_phys/|Ω| is unique among symmetry-respecting
      measures, not arbitrary.

(5) Holographic boundary→bulk conversion:
      The horizon set H (256 states) has a 1-step neighborhood under all bytes
      that covers the full bulk Ω exactly (size 65,536).

Finally, we compute:
      CSM = N_phys / |Ω|
and show the UHI demand margin.

This file intentionally DOES NOT re-test:
- pack/unpack correctness
- byte inverse / commutator translation
- Walsh/MacWilliams duality
- full closed-form step law
Those are already certified elsewhere (tests/physics/test_physics_*.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests._physics_utils import table
from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_B12,
    LAYER_MASK_12,
    mask12_for_byte,
)
from src.router.kernel import RouterKernel


# --- Physical constants ---
# Import canonical constants from production code
from src.app.coordination import ATOMIC_HZ_CS133, OMEGA_SIZE

SPEED_OF_LIGHT = 299_792_458  # m/s (used only for c-cancellation stress test)
SECONDS_PER_YEAR = 365 * 24 * 10 * 10  # Used for coverage calculations

# --- Router constants ---
CODE_SIZE = 256
HORIZON_SIZE = 256  # |H| = 2^8

# --- Moments Economy parameters (used only for margin check) ---
WORLD_POP = 8_100_000_000
UHI_PER_YEAR_MU = 87_600  # 240 MU/day × 365
REQUIRED_CAPACITY_PER_YEAR = WORLD_POP * UHI_PER_YEAR_MU


# =============================================================================
# Fixtures / helpers
# =============================================================================

@pytest.fixture(scope="module")
def kernel():
    atlas_dir = Path("data/atlas")
    if not atlas_dir.exists():
        pytest.skip("Atlas not built. Run: python -m src.router.atlas")
    return RouterKernel(atlas_dir)


def _mask_code_set() -> set[int]:
    """C = {mask12_for_byte(b): b in 0..255}"""
    return {mask12_for_byte(b) & 0xFFF for b in range(256)}




# =============================================================================
# Test 1: Physical microcell count and c-cancellation (stress)
# =============================================================================

def test_physical_microcell_count_closed_form_and_c_cancellation():
    """
    Establish raw physical microcell count N_phys for the 1-second causal container:

      V_1s = (4/3)π (c·1s)^3
      λ_Cs = c / f_Cs
      N_phys = V_1s / λ_Cs^3 = (4/3)π f_Cs^3

    We also stress-test c-cancellation:
      Replace c -> k*c in both V and λ, N_phys must stay invariant.
    """
    f = ATOMIC_HZ_CS133

    def N_phys_from(c: float) -> float:
        R = c * 1.0
        V = (4.0 / 3.0) * math.pi * (R**3)
        lam = c / f
        return V / (lam**3)

    # direct closed form
    N_closed = (4.0 / 3.0) * math.pi * (f**3)

    # compute with real c and scaled c values
    N1 = N_phys_from(SPEED_OF_LIGHT)
    N2 = N_phys_from(2.0 * SPEED_OF_LIGHT)
    N3 = N_phys_from(0.1 * SPEED_OF_LIGHT)

    rel1 = abs(N1 - N_closed) / N_closed
    rel2 = abs(N2 - N_closed) / N_closed
    rel3 = abs(N3 - N_closed) / N_closed

    table(
        "PHYSICAL MICROCELL COUNT (c cancels)",
        [
            ("f_Cs (Hz)", f"{f:,}"),
            ("N_phys closed = (4/3)π f^3", f"{N_closed:.6e}"),
            ("N_phys(c)", f"{N1:.6e}  (rel err {rel1:.3e})"),
            ("N_phys(2c)", f"{N2:.6e} (rel err {rel2:.3e})"),
            ("N_phys(0.1c)", f"{N3:.6e} (rel err {rel3:.3e})"),
        ],
    )

    assert rel1 < 1e-14
    assert rel2 < 1e-14
    assert rel3 < 1e-14


# =============================================================================
# Test 2: Ω = C × C in (u,v) coordinates (conversion lattice)
# =============================================================================

def test_router_omega_is_cartesian_product_CxC(kernel: RouterKernel):
    """
    Certify the conversion lattice claim:

      In mask coordinates (u,v) relative to archetype,
      Ω is exactly C × C where |C|=256, |Ω|=256^2.

    This test extracts u_set and v_set from the REAL ontology and verifies:
      - |u_set| = |v_set| = 256
      - u_set == v_set == C (mask code from bytes)
    """
    ont = kernel.ontology.astype(np.uint32)
    assert int(ont.size) == OMEGA_SIZE

    # Extract A,B arrays and compute u,v arrays
    A = ((ont >> 12) & 0xFFF).astype(np.uint16)
    B = (ont & 0xFFF).astype(np.uint16)

    u = (A ^ np.uint16(ARCHETYPE_A12)).astype(np.uint16)
    v = (B ^ np.uint16(ARCHETYPE_B12)).astype(np.uint16)

    u_set = set(int(x) for x in np.unique(u))
    v_set = set(int(x) for x in np.unique(v))
    C = _mask_code_set()

    table(
        "Ω AS CONVERSION LATTICE: Ω = C × C",
        [
            ("|Ω|", f"{int(ont.size):,}"),
            ("|u_set|", f"{len(u_set)}"),
            ("|v_set|", f"{len(v_set)}"),
            ("|C|", f"{len(C)}"),
            ("u_set == C", str(u_set == C)),
            ("v_set == C", str(v_set == C)),
        ],
    )

    assert len(u_set) == CODE_SIZE
    assert len(v_set) == CODE_SIZE
    assert u_set == C
    assert v_set == C


# =============================================================================
# Test 3: Strong isotropy inside Ω via uniform d = u⊕v distribution
# =============================================================================

def test_difference_distribution_is_exactly_uniform_over_C(kernel: RouterKernel):
    """
    Strong, exact statement:

      Let d = u XOR v where (u,v) are mask coordinates of Ω.
      Then across all 65,536 states, d is uniformly distributed over C:

        For every d in C: count(d) = 256
        For every d not in C: count(d) = 0

    This is the exact "no privileged direction" conversion statement inside Ω
    and is more relevant than approximate isotropy heuristics.
    """
    ont = kernel.ontology.astype(np.uint32)

    A = ((ont >> 12) & 0xFFF).astype(np.uint16)
    B = (ont & 0xFFF).astype(np.uint16)
    u = (A ^ np.uint16(ARCHETYPE_A12)).astype(np.uint16)
    v = (B ^ np.uint16(ARCHETYPE_B12)).astype(np.uint16)

    d = (u ^ v).astype(np.uint16)  # 12-bit values

    # Count occurrences in 0..4095
    counts = np.bincount(d.astype(np.int64), minlength=4096).astype(np.int64)

    C = _mask_code_set()
    nonzero_idxs = set(int(i) for i in np.where(counts != 0)[0])

    # All nonzero counts must be exactly 256
    bad_counts = [(i, int(counts[i])) for i in nonzero_idxs if int(counts[i]) != 256]

    table(
        "UNIFORMITY: d = u XOR v distribution over Ω",
        [
            ("Nonzero d values", f"{len(nonzero_idxs)}"),
            ("Expected |C|", f"{len(C)}"),
            ("Support equals C", str(nonzero_idxs == C)),
            ("All nonzero counts == 256", str(len(bad_counts) == 0)),
            ("Example bad counts", str(bad_counts[:5]) if bad_counts else "[]"),
        ],
    )

    assert nonzero_idxs == C
    assert len(bad_counts) == 0


# =============================================================================
# Test 4: Regular action of 2-byte words from ANY start state (bijection)
# =============================================================================

def test_two_byte_words_form_bijection_to_omega_from_any_start(kernel: RouterKernel):
    """
    This is the core "measure forcing" test for Moments Economy:

    For a fixed start state index s, consider all 2-byte words (x,y) with x,y in 0..255.
    Compute:

        out(x,y) = T_y(T_x(s))

    Claim: the map (x,y) -> out is a bijection onto Ω.
    Equivalently: the set {out(x,y): x,y ∈ bytes} has size 65,536.

    We certify this for multiple starting states to demonstrate:
      - There is no privileged origin (CS is not "a bit" or "a location").
      - The even-word subgroup acts regularly (free + transitive).

    ASSUMPTION (semantic, not mathematical):
      The physical microcell capacity should be distributed over Router states using
      a measure that is invariant under the Router's symmetries (the action of the
      byte group / even-word subgroup).

    Given the transitive action (proven above), this assumption forces uniformity:
      any symmetry-invariant measure over Ω must be uniform.
      Therefore: per-state share = N_phys/|Ω| is unique among symmetry-respecting
      measures, not arbitrary.
    """
    epi = kernel.epistemology
    n = int(kernel.ontology.size)
    assert n == OMEGA_SIZE

    bytes_arr = np.arange(256, dtype=np.int64)

    # choose representative start states: archetype, mid, last, and a few random
    rng = np.random.default_rng(20260113)
    starts = [kernel.archetype_index, n // 2, n - 1] + rng.integers(0, n, size=3, dtype=np.int64).tolist()

    for s_idx in starts:
        s_idx = int(s_idx)
        # Step 1: apply x to start (256 outputs)
        i1 = epi[s_idx, bytes_arr].astype(np.int64)  # (256,)
        # Step 2: apply y to each i1 (256×256 outputs)
        i2 = epi[i1[:, None], bytes_arr[None, :]].astype(np.int64)  # (256,256)
        flat = i2.reshape(-1)
        uniq = np.unique(flat)

        table(
            "2-BYTE BIJECTION CHECK",
            [
                ("start_index", str(s_idx)),
                ("unique outputs", f"{uniq.size:,}"),
                ("expected |Ω|", f"{OMEGA_SIZE:,}"),
                ("bijective", str(int(uniq.size) == OMEGA_SIZE)),
            ],
        )

        assert int(uniq.size) == OMEGA_SIZE


# =============================================================================
# Test 5: Holographic boundary-to-bulk conversion (H -> Ω in one step)
# =============================================================================

def test_horizon_one_step_neighborhood_covers_full_bulk(kernel: RouterKernel):
    """
    Holographic conversion property relevant to "Moment as container":

    Horizon set H is defined by A = ~B (fixed points of byte 0xAA in Ω).
    |H| must be 256.

    Claim: the one-step neighborhood of H under all 256 bytes covers Ω:
      {T_b(h): h∈H, b∈bytes} = Ω

    This is the exact boundary→bulk encoding property.
    """
    ont = kernel.ontology.astype(np.uint32)
    epi = kernel.epistemology

    A = ((ont >> 12) & 0xFFF).astype(np.uint16)
    B = (ont & 0xFFF).astype(np.uint16)
    horizon_mask = (A == (B ^ np.uint16(LAYER_MASK_12)))
    horizon_idxs = np.where(horizon_mask)[0].astype(np.int64)

    assert int(horizon_idxs.size) == HORIZON_SIZE

    bytes_arr = np.arange(256, dtype=np.int64)
    next_idxs = epi[horizon_idxs[:, None], bytes_arr[None, :]].astype(np.int64).reshape(-1)
    uniq = np.unique(next_idxs)

    table(
        "HOLOGRAPHIC COVERAGE: H -> Ω in one step",
        [
            ("|H|", f"{int(horizon_idxs.size)}"),
            ("unique next states", f"{int(uniq.size):,}"),
            ("expected |Ω|", f"{OMEGA_SIZE:,}"),
            ("covers full Ω", str(int(uniq.size) == OMEGA_SIZE)),
        ],
    )

    assert int(uniq.size) == OMEGA_SIZE


# =============================================================================
# Final: CSM capacity and UHI margin (no extra physics claims)
# =============================================================================

def test_csm_capacity_and_uhi_margin():
    """
    Given the conversion lattice results above, define the capacity standard:

      N_phys = (4/3)π f_Cs^3  (raw physical microcells in 1-second causal container)
      CSM    = N_phys / |Ω|   (uniform coarse-grain per Router state)

    The uniform division by |Ω| is not arbitrary. It is forced by requiring
    the measure to be invariant under the Router's symmetries (byte group action).
    Given the transitive action (proven in test_regular_action_2_bytes), uniformity
    is the unique symmetry-invariant measure. This is "unique among symmetry-respecting
    measures", not "unique in all possible mappings."

    CRITICAL: CSM is a FIXED TOTAL CAPACITY, not a rate. The "1 second" is consumed
    in the derivation of N_phys. We calculate how many years this fixed total can
    cover global UHI demand.
    """
    f = ATOMIC_HZ_CS133
    N_phys = (4.0 / 3.0) * math.pi * (f**3)

    CSM = N_phys / float(OMEGA_SIZE)  # Total capacity in MU

    required_per_year = float(REQUIRED_CAPACITY_PER_YEAR)
    coverage_years = CSM / required_per_year

    table(
        "CSM CAPACITY (conversion result) and UHI coverage",
        [
            ("N_phys", f"{N_phys:.6e}"),
            ("|Ω|", f"{OMEGA_SIZE:,}"),
            ("CSM (total capacity)", f"{CSM:.6e}"),
            ("UHI required/year", f"{required_per_year:.6e}"),
            ("Coverage (years)", f"{coverage_years:.6e}"),
        ],
    )

    assert CSM > 1e24
    assert coverage_years > 1e10  # Should be ~7e10 years (70 billion years)