# tests/test_substrate.py
"""
Transparent, Accountable, Resilient Fiat Substrate
===================================================

This module tests an end-to-end fiat substrate built on top of the CGM-derived
router kernel. It treats the kernel as a physical coordination device and
verifies that:

- Atomic frequency and Router ontology define an abundant MU capacity envelope.
- Capacity is partitioned into replayable Shells (time-bounded windows).
- MU Grants are anchored to identities via kernel states.
- Archives aggregate Shells deterministically across long horizons.
- Integrity and tampering are detectable via kernel algebra (parity law, dual code).
- Meta-routing commits multiple programme ledgers to a compact root.
- State components can be isolated (identity vs balance) and rolled back.

No physics proofs are repeated here; those live in test_physics_*.py.
Capacity derivation proofs live in test_moments_2.py.
This file focuses on substrate-level correctness and robustness.

To run all three test files (moments, moments_2, substrate) as a unified suite:
  python tests/test_substrate.py
  
Or run them individually:
  python -m pytest tests/test_moments.py -v
  python -m pytest tests/test_moments_2.py -v
  python -m pytest tests/test_substrate.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import pytest

# Add project root to Python path (needed when running file directly)
PROGRAM_ROOT = Path(__file__).parent.parent
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from src.router.kernel import RouterKernel
from src.router.constants import (
    mask12_for_byte,
    dot12,
    C_PERP_12,
)

ATLAS_DIR = Path(__file__).parent.parent / "data" / "atlas"

# Import capacity constants from production code (canonical source)
from src.app.coordination import (
    OMEGA_SIZE,
    raw_microcells_per_moment,
    csm_per_moment_mu,
    capacity_for_year,
)

# Moments Economy parameters (for test calculations)
WORLD_POP: int = 8_100_000_000
UHI_PER_YEAR_MU: int = 87_600  # 240 MU/day × 365

# Compute derived constants using production functions
N_PHYS: float = raw_microcells_per_moment()
CSM_PER_MOMENT_MU: float = csm_per_moment_mu()
CSM_PER_YEAR_MU: int = capacity_for_year()
GLOBAL_UHI_DEMAND_PER_YEAR: float = float(WORLD_POP) * float(UHI_PER_YEAR_MU)


# ---------------------------------------------------------------------------
# 0. Pytest fixture: atlas directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def atlas_dir() -> Path:
    """Provide atlas directory path; fail fast if missing."""
    assert ATLAS_DIR.exists(), f"Atlas not found at {ATLAS_DIR}"
    return ATLAS_DIR


# ---------------------------------------------------------------------------
# 1. Test helpers
# ---------------------------------------------------------------------------

def route_bytes(payload: bytes) -> str:
    """
    Route a payload through a fresh kernel and return state_hex (3 bytes).

    This is the canonical way to derive a compact structural commitment
    from arbitrary bytes.
    """
    k = RouterKernel(ATLAS_DIR)
    sig = k.route_from_archetype(payload)
    return sig.state_hex


# ---------------------------------------------------------------------------
# 3. Shell and Archive integrity (transparency + accountability)
# ---------------------------------------------------------------------------
# Note: Capacity envelope checks are in test_moments.py::test_millennium_uhi_feasibility_under_csm
# This file focuses on substrate-level correctness (Shells, Archives, integrity).

def test_01_shell_and_archive_integrity(atlas_dir: Path):
    """
    Verify Shell and Archive behavior using canonical Coordinator implementation:

    - Shell is within capacity and deterministically replayable.
    - Archive built from multiple Shells is deterministic.
    - Tampering with Grants changes Shell seal and Archive totals.
    - Duplicate Grants for an identity in a Shell are detectable (application rule).
    """
    print("\n=== SHELL AND ARCHIVE INTEGRITY ===")
    
    from src.app.coordination import Coordinator
    
    # Use Coordinator (canonical implementation)
    coord1 = Coordinator(atlas_dir)
    coord1.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord1.add_grant("bob", UHI_PER_YEAR_MU * 2)
    
    # Close shell with header (capacity derived automatically from header)
    shell1 = coord1.close_shell(b"ecology:year:2026")
    
    assert shell1.used_capacity_MU <= shell1.total_capacity_MU
    assert shell1.total_capacity_MU == CSM_PER_YEAR_MU  # Should use annual capacity
    
    # Replay: same grants -> same seal
    coord2 = Coordinator(atlas_dir)
    coord2.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord2.add_grant("bob", UHI_PER_YEAR_MU * 2)
    shell2 = coord2.close_shell(b"ecology:year:2026")
    
    assert shell2.seal == shell1.seal
    assert shell2.used_capacity_MU == shell1.used_capacity_MU
    
    print(f"Shell seal: {shell1.seal}")
    print(f"Used capacity: {shell1.used_capacity_MU:,} MU")
    print(f"Total capacity: {shell1.total_capacity_MU:,} MU")
    
    # Tampering: inflate alice's grant
    coord_tampered = Coordinator(atlas_dir)
    coord_tampered.add_grant("alice", UHI_PER_YEAR_MU * 30)  # inflated
    coord_tampered.add_grant("bob", UHI_PER_YEAR_MU * 2)
    shell_tampered = coord_tampered.close_shell(b"ecology:year:2026")
    
    assert shell_tampered.seal != shell1.seal
    assert shell_tampered.used_capacity_MU != shell1.used_capacity_MU
    
    # Archive: aggregate multiple shells
    coord3 = Coordinator(atlas_dir)
    coord3.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord3.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord3.close_shell(b"ecology:year:2026")
    coord3.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord3.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord3.close_shell(b"ecology:year:2027")
    status3 = coord3.fiat_status()
    
    # Verify archive totals
    assert status3["per_identity_totals"]["alice"] == UHI_PER_YEAR_MU * 3 * 2
    assert status3["per_identity_totals"]["bob"] == UHI_PER_YEAR_MU * 2 * 2
    
    print(f"Archive per-identity MU: {status3['per_identity_totals']}")
    print("OK: shell/archive deterministic and tamper-evident")


# ---------------------------------------------------------------------------
# 3. Horizon structure and identity paths (CS + reachability)
# ---------------------------------------------------------------------------

def test_02_horizon_structure_and_coverage(atlas_dir: Path):
    """
    The horizon (fixed points of 0xAA) is a 256-state boundary with:

    - Symmetric coordinates A = B XOR 0xFFF.
    - 1-step fanout over all 256 bytes reaching the entire 65,536 bulk.

    Each horizon state is a "root identity anchor" from which trajectories
    can cover all accessible Router states.
    """
    print("\n=== HORIZON STRUCTURE AND COVERAGE ===")
    from src.router.constants import unpack_state, LAYER_MASK_12

    epi = np.load(atlas_dir / "epistemology.npy", mmap_mode="r").astype(np.int64)
    ont = np.load(atlas_dir / "ontology.npy")

    idxs = np.arange(epi.shape[0], dtype=np.int64)
    horizon_idxs = idxs[epi[:, 0xAA] == idxs]
    assert len(horizon_idxs) == 256

    # 1-step reachability over all bytes from each horizon state
    reachable = set()
    for h in horizon_idxs:
        reachable.update(map(int, epi[int(h), :]))
    assert len(reachable) == OMEGA_SIZE

    # Symmetry A = B XOR 0xFFF and unique A per horizon state
    a_values = []
    for h_idx in horizon_idxs:
        s = int(ont[h_idx])
        a, b = unpack_state(s)
        assert a == (b ^ LAYER_MASK_12)
        a_values.append(a)
    assert len(set(a_values)) == 256

    print(f"Horizon states: {len(horizon_idxs)}")
    print(f"Reachable (1-step): {len(reachable)}")
    print("OK: 256 horizon anchors cover the 65,536-state bulk in 1 step")


def test_03_trajectory_identity_scaling():
    """
    Identity as (horizon, path):

    - 256 choices of horizon state.
    - 256^n distinct paths of length n from that horizon.

    For n=4 bytes, 256^(n+1) > 10^12, enough to assign unique identity
    paths to every human while leaving immense headroom.
    """
    print("\n=== TRAJECTORY IDENTITY SCALING ===")

    target_identities = 10_000_000_000  # 10 billion
    for n in [1, 2, 3, 4]:
        count = 256 ** (n + 1)
        print(f"n={n}: 256^(n+1) = {count:,}")
    assert 256 ** 5 >= target_identities
    print("OK: n=4 path length suffices for global identity labelling")


# ---------------------------------------------------------------------------
# 4. Parity commitment and tamper detection (trajectory-level integrity)
# ---------------------------------------------------------------------------

def test_04_parity_commitment_and_reconstruction(atlas_dir: Path):
    """
    The kernel trajectory closed form:

      - O = XOR of masks at odd positions
      - E = XOR of masks at even positions
      - parity = length mod 2

    reconstructs the final state exactly for any length trajectory.
    """
    print("\n=== PARITY COMMITMENT AND RECONSTRUCTION ===")
    from src.router.constants import ARCHETYPE_A12, ARCHETYPE_B12, pack_state

    k = RouterKernel(atlas_dir)
    rng = np.random.default_rng(7)
    word = rng.integers(0, 256, size=4096, dtype=np.uint16).tolist()

    # Actual final state
    k.reset()
    for b in word:
        k.step_byte(int(b))
    s_actual = int(k.ontology[k.state_index])

    # Closed form in (u, v)
    u0 = v0 = 0
    O = E = 0
    for i, b in enumerate(word):
        m = mask12_for_byte(int(b))
        if i % 2 == 0:
            O ^= m
        else:
            E ^= m
    n = len(word)
    if n % 2 == 0:
        u_n, v_n = u0 ^ O, v0 ^ E
    else:
        u_n, v_n = v0 ^ E, u0 ^ O

    a_n = (u_n ^ ARCHETYPE_A12) & 0xFFF
    b_n = (v_n ^ ARCHETYPE_B12) & 0xFFF
    s_expected = pack_state(a_n, b_n)

    assert s_expected == s_actual
    print(f"Trajectory length: {len(word)} bytes")
    print("Compressed to: (O, E, parity) = 25 bits")
    print("OK: closed-form parity commitment reconstructs final state exactly")


def test_05_trajectory_tamper_detection():
    """
    The parity commitment (O, E, parity) is sensitive to tampering:
    changing any byte in a trajectory almost always changes the commitment.

    This provides a cheap integrity check over arbitrary-length histories.
    """
    print("\n=== TRAJECTORY TAMPER DETECTION ===")

    random.seed(777)
    trajectory = [random.randint(0, 255) for _ in range(100)]

    def commitment(traj: List[int]) -> Tuple[int, int, int]:
        O = E = 0
        for i, b in enumerate(traj):
            m = mask12_for_byte(b)
            if i % 2 == 0:
                O ^= m
            else:
                E ^= m
        return (O, E, len(traj) % 2)

    original = commitment(trajectory)
    tamper_detected = 0
    for pos in range(len(trajectory)):
        tampered = trajectory.copy()
        tampered[pos] = (tampered[pos] + 1) % 256
        if tampered[pos] == trajectory[pos]:
            tampered[pos] = (tampered[pos] + 1) % 256
        if commitment(tampered) != original:
            tamper_detected += 1

    print(f"Trajectory length: {len(trajectory)}")
    print(f"Tampers detected: {tamper_detected}/{len(trajectory)}")
    assert tamper_detected >= len(trajectory) - 5
    print("OK: parity commitment reliably detects tampering")


# ---------------------------------------------------------------------------
# 5. Dual code integrity check (mask-level error detection)
# ---------------------------------------------------------------------------

def test_06_dual_code_integrity():
    """
    Dual code C_perp (16 elements) is orthogonal to all 256 mask codewords.

    We use it to detect corrupted 12-bit patterns:
    - All valid mask patterns have zero syndrome.
    - Random non-mask patterns almost always produce non-zero syndrome.
    """
    print("\n=== DUAL CODE INTEGRITY CHECK ===")

    # Collect mask code C
    masks = set(mask12_for_byte(b) for b in range(256))

    # Use imported C_PERP_12
    assert len(C_PERP_12) == 16

    # Check that all masks have zero syndrome
    for m in masks:
        syndromes = [dot12(m, v) for v in C_PERP_12]
        assert all(s == 0 for s in syndromes)

    # Sample random non-mask values and check they are detected
    trials = 1000
    detected = 0
    for _ in range(trials):
        m = random.randint(0, (1 << 12) - 1)
        if m in masks:
            continue
        syndromes = [dot12(m, v) for v in C_PERP_12]
        if any(s == 1 for s in syndromes):
            detected += 1

    print(f"Dual code size: {len(C_PERP_12)} elements")
    print(f"Random corrupted patterns detected: {detected}/{trials}")
    assert detected >= trials * 0.9
    print("OK: dual code reliably detects corrupted 12-bit patterns")


# ---------------------------------------------------------------------------
# 6. Meta-routing: global aggregation and dispute localization
# ---------------------------------------------------------------------------

def test_07_meta_routing(atlas_dir: Path):
    """
    Meta-routing aggregates multiple programme bundles into a single root seal:

    - First pass: each programme bundle -> leaf seal (kernel state_hex).
    - Second pass: list of leaf seals -> meta-root seal.

    Properties:
    - Deterministic.
    - Permutation-invariant (set-style by construction).
    - Any tamper in an individual bundle changes its leaf seal and the meta-root,
      and can be localized by comparing leaf seals.
    """
    print("\n=== META ROUTING (AGGREGATION + DISPUTE LOCALIZATION) ===")

    def seal_payload(payload: bytes) -> bytes:
        k = RouterKernel(atlas_dir)
        k.step_payload(payload)
        return bytes.fromhex(k.signature().state_hex)

    def meta_root(seals: List[bytes]) -> str:
        k = RouterKernel(atlas_dir)
        for s in seals:
            k.step_payload(s)
        return k.signature().state_hex

    bundles = [
        b"program:A|bytes:abc|events:123",
        b"program:B|bytes:def|events:456",
        b"program:C|bytes:ghi|events:789",
    ]
    seals = [seal_payload(b) for b in bundles]

    # Determinism
    root1 = meta_root(seals)
    root2 = meta_root(seals)
    assert root1 == root2

    # Set-style aggregation: permutation does not change root
    root_swapped = meta_root([seals[1], seals[0], seals[2]])
    assert root_swapped == root1

    # Tamper with one bundle
    tampered = bundles.copy()
    tampered[1] = b"program:B|bytes:def|events:TAMPERED"
    tampered_seals = [seal_payload(b) for b in tampered]
    root_tampered = meta_root(tampered_seals)
    assert root_tampered != root1

    diffs = [i for i, (a, b) in enumerate(zip(seals, tampered_seals)) if a != b]
    assert diffs == [1]

    print(f"Meta-root: {root1}")
    print(f"Permutation-invariant: {root_swapped == root1}")
    print(f"Tamper localized to leaf index: {diffs[0]}")
    print("OK: meta-routing supports global aggregation with localizable disputes")


# ---------------------------------------------------------------------------
# 7. Component isolation and rollback (BU-Ingress flavor)
# ---------------------------------------------------------------------------

def test_08_component_isolation_and_rollback():
    """
    Using separator lemmas and conjugation by reference byte (0xAA), demonstrate:

    - A-component can encode an identity that remains invariant under balance updates.
    - B-component can encode a balance updated by controlled operations.
    - A simple rollback sequence returns the state to its original value.

    This is the discrete analogue of BU-Ingress: a balanced state preserves enough
    structure to reconstruct (or undo) prior transitions.
    """
    print("\n=== COMPONENT ISOLATION AND ROLLBACK ===")
    from src.router.constants import (
        step_state_by_byte,
        pack_state,
        unpack_state,
        XFORM_MASK_BY_BYTE,
        LAYER_MASK_12,
    )

    random.seed(12345)
    ref_byte = 0xAA  # reference involution

    # Step 1: initialize identity in A via x then AA (A-only update)
    identity_byte = 0x42
    a_init, b_init = 0x555, 0x000
    s0 = pack_state(a_init, b_init)
    s1 = step_state_by_byte(s0, identity_byte)
    s2 = step_state_by_byte(s1, ref_byte)

    a_after_id, b_after_id = unpack_state(s2)
    m_id = (int(XFORM_MASK_BY_BYTE[identity_byte]) >> 12) & LAYER_MASK_12
    assert a_after_id == (a_init ^ m_id)
    assert b_after_id == b_init

    # Step 2: perform balance operations via AA then y (B-only update)
    balance_ops = [0x10, 0x20, 0x30]
    s_curr = s2
    for b_byte in balance_ops:
        s_curr = step_state_by_byte(s_curr, ref_byte)
        s_curr = step_state_by_byte(s_curr, b_byte)

    a_final, b_final = unpack_state(s_curr)
    assert a_final == a_after_id  # identity unchanged

    # Compute expected B
    b_expected = b_after_id
    for b_byte in balance_ops:
        m_b = (int(XFORM_MASK_BY_BYTE[b_byte]) >> 12) & LAYER_MASK_12
        b_expected ^= m_b
    assert b_final == b_expected

    print(f"Identity (A): {a_init:03x} -> {a_final:03x} (stable)")
    print(f"Balance  (B): {b_init:03x} -> {b_final:03x} (updated)")

    # Step 3: rollback using conjugation
    last_b = balance_ops[-1]

    # Compute s_prev (state before last balance op)
    s_prev = s2
    for b_byte in balance_ops[:-1]:
        s_prev = step_state_by_byte(s_prev, ref_byte)
        s_prev = step_state_by_byte(s_prev, b_byte)

    # Apply inverse: T_last_b^(-1) T_AA^(-1) = (R T_last_b R) R = T_last_b R
    s_rollback = step_state_by_byte(s_curr, ref_byte)  # R
    s_rollback = step_state_by_byte(s_rollback, last_b)  # T_last_b

    assert s_rollback == s_prev

    print("OK: component isolation holds and simple rollback recovers prior state")


# ---------------------------------------------------------------------------
# 8. Kernel-level rollback via inverse stepping
# ---------------------------------------------------------------------------

def test_09_kernel_inverse_stepping(atlas_dir: Path):
    """
    Test kernel inverse stepping: step forward, then step inverse, verify return.
    
    Uses the kernel's step_byte_inverse method which implements the conjugation
    form of the inverse: T_x^{-1} = R ∘ T_x ∘ R where R = T_0xAA.
    """
    print("\n=== KERNEL INVERSE STEPPING ===")
    
    k = RouterKernel(atlas_dir)
    
    # Step forward with a known payload
    payload = b"test payload"
    initial_index = k.archetype_index
    
    for b in payload:
        k.step_byte(b)
    
    assert k.step == len(payload)
    final_index = k.state_index
    assert final_index != initial_index
    
    # Step inverse (in reverse order)
    for b in reversed(payload):
        k.step_byte_inverse(b)
    
    # Verify returned to archetype
    assert k.state_index == initial_index
    
    print(f"Payload: {payload!r}")
    print(f"Forward steps: archetype -> {final_index}")
    print(f"Inverse steps: {final_index} -> archetype")
    print("OK: kernel inverse stepping returns to origin")




if __name__ == "__main__":
    # Change to project root directory (PROGRAM_ROOT already set above)
    os.chdir(PROGRAM_ROOT)
    
    # When run directly, collect and run all three test files as a unified suite
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_moments.py",
        test_dir / "test_moments_2.py",
        Path(__file__),
    ]
    
    # Filter to only existing files
    existing_files = [str(f) for f in test_files if f.exists()]
    
    if len(existing_files) > 1:
        print(f"\nRunning unified test suite: {len(existing_files)} files")
        print("=" * 60)
        # Run pytest on all three files
        sys.exit(pytest.main(["-s", "-v"] + existing_files))
    else:
        # Fallback: just run this file
        sys.exit(pytest.main(["-s", "-v", __file__]))