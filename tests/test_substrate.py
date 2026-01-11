# tests/test_substrate.py
"""
Transparent, Accountable, Resilient Fiat Substrate
===================================================

This module tests an end-to-end fiat substrate built on top of the CGM-derived
router kernel. It treats the kernel as a physical coordination device and
verifies that:

- Atomic and kernel throughput define an abundant MU capacity envelope.
- Capacity is partitioned into replayable Shells (time-bounded windows).
- MU Grants are anchored to identities via kernel states.
- Archives aggregate Shells deterministically across long horizons.
- Integrity and tampering are detectable via kernel algebra (parity law, dual code).
- Meta-routing commits multiple programme ledgers to a compact root.
- State components can be isolated (identity vs balance) and rolled back.

No physics proofs are repeated here; those live in test_physics_*.py.
This file focuses on substrate-level correctness and robustness.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import pytest

from src.router.kernel import RouterKernel
from src.router.constants import (
    mask12_for_byte,
    dot12,
    C_PERP_12,
)
from src.app.coordination import Coordinator, CAPACITY_PER_YEAR_MU

ATLAS_DIR = Path(__file__).parent.parent / "data" / "atlas"


# ---------------------------------------------------------------------------
# 0. Pytest fixture: atlas directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def atlas_dir() -> Path:
    """Provide atlas directory path; fail fast if missing."""
    assert ATLAS_DIR.exists(), f"Atlas not found at {ATLAS_DIR}"
    return ATLAS_DIR


# ---------------------------------------------------------------------------
# 1. Physical and economic constants (from Moments Economy)
# ---------------------------------------------------------------------------

# World population (approximate, for scaling checks)
WORLD_POP = 8_100_000_000

# Unconditional High Income: 4 hours/day at 60 MU/hour = 240 MU/day
UHI_PER_YEAR_MU = 87_600

# CAPACITY_PER_YEAR_MU imported from src.app.coordination


# ---------------------------------------------------------------------------
# 2. Test helpers (using runtime implementations)
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
# 4. Capacity envelope for UHI over realistic horizons
# ---------------------------------------------------------------------------

def test_01_capacity_envelope_for_uhi():
    """
    One year and one millennium of UHI for the world fit comfortably
    inside the conservative MU capacity envelope.

    This shows that at planetary and millennial scales, capacity is solely
    a governance question, not a physical shortage.
    """
    print("\n=== CAPACITY ENVELOPE VS UHI ===")

    total_uhi_year = WORLD_POP * UHI_PER_YEAR_MU
    usage_fraction_year = total_uhi_year / CAPACITY_PER_YEAR_MU

    years = 1000
    total_uhi_mill = total_uhi_year * years
    capacity_mill = CAPACITY_PER_YEAR_MU * years
    usage_fraction_mill = total_uhi_mill / capacity_mill

    print(f"Capacity/year (MU): {CAPACITY_PER_YEAR_MU:,}")
    print(f"UHI/year (MU):      {total_uhi_year:,}")
    print(f"Usage/year:         {usage_fraction_year:.12e}")
    print(f"Capacity/{years}y (MU): {capacity_mill:,}")
    print(f"UHI/{years}y (MU):      {total_uhi_mill:,}")
    print(f"Usage/{years}y:         {usage_fraction_mill:.12e}")

    # Conservative safety margins
    assert usage_fraction_year < 1e-6
    assert usage_fraction_mill < 1e-3


# ---------------------------------------------------------------------------
# 5. Shell and Archive integrity (transparency + accountability)
# ---------------------------------------------------------------------------

def test_02_shell_and_archive_integrity():
    """
    Verify Shell and Archive behavior:

    - Shell is within capacity and deterministically replayable.
    - Archive built from multiple Shells is deterministic.
    - Tampering with Grants changes Shell seal and Archive totals.
    - Duplicate Grants for an identity in a Shell are detectable (application rule).
    """
    print("\n=== SHELL AND ARCHIVE INTEGRITY ===")

    header = b"ecology:year:2026"
    capacity = CAPACITY_PER_YEAR_MU

    # Use Coordinator for shell creation
    coord1 = Coordinator(ATLAS_DIR)
    coord1.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord1.add_grant("bob", UHI_PER_YEAR_MU * 2)
    shell = coord1.close_shell(header, capacity)
    
    assert shell.used_capacity_MU <= shell.total_capacity_MU
    
    # Replay: create same shell with another Coordinator
    coord2 = Coordinator(ATLAS_DIR)
    coord2.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord2.add_grant("bob", UHI_PER_YEAR_MU * 2)
    shell2 = coord2.close_shell(header, capacity)
    
    assert shell2.seal == shell.seal
    assert shell2.used_capacity_MU == shell.used_capacity_MU

    # Archive determinism across shells
    coord3 = Coordinator(ATLAS_DIR)
    coord3.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord3.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord3.close_shell(b"ecology:year:2026", capacity)
    coord3.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord3.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord3.close_shell(b"ecology:year:2027", capacity)
    status1 = coord3.fiat_status()
    
    coord4 = Coordinator(ATLAS_DIR)
    coord4.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord4.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord4.close_shell(b"ecology:year:2026", capacity)
    coord4.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord4.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord4.close_shell(b"ecology:year:2027", capacity)
    status2 = coord4.fiat_status()
    
    assert status1["per_identity_totals"] == status2["per_identity_totals"]
    assert status1["used_capacity_MU"] == status2["used_capacity_MU"]

    print(f"Shell seal: {shell.seal}")
    print(f"Archive per-identity MU: {status1['per_identity_totals']}")

    # Tampering: inflate alice's grant
    coord_tampered = Coordinator(ATLAS_DIR)
    coord_tampered.add_grant("alice", UHI_PER_YEAR_MU * 30)
    coord_tampered.add_grant("bob", UHI_PER_YEAR_MU * 2)
    shell_tampered = coord_tampered.close_shell(header, capacity)
    status_tampered = coord_tampered.fiat_status()

    assert shell_tampered.seal != shell.seal
    assert status_tampered["per_identity_totals"]["alice"] != status1["per_identity_totals"]["alice"]

    # Duplicate grant detection (application-level rule)
    coord_dup = Coordinator(ATLAS_DIR)
    coord_dup.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord_dup.add_grant("bob", UHI_PER_YEAR_MU * 2)
    try:
        coord_dup.add_grant("alice", UHI_PER_YEAR_MU)
        assert False, "Expected ValueError for duplicate grant"
    except ValueError:
        pass  # Expected

    print("OK: shell/archive deterministic, tamper-evident, and duplicate-detectable")


# ---------------------------------------------------------------------------
# 6. Horizon structure and identity paths (CS + reachability)
# ---------------------------------------------------------------------------

def test_03_horizon_structure_and_coverage(atlas_dir: Path):
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
    assert len(reachable) == 65_536

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


def test_04_trajectory_identity_scaling():
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
# 7. Parity commitment and tamper detection (trajectory-level integrity)
# ---------------------------------------------------------------------------

def test_05_parity_commitment_and_reconstruction(atlas_dir: Path):
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


def test_06_trajectory_tamper_detection():
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
# 8. Dual code integrity check (mask-level error detection)
# ---------------------------------------------------------------------------

def test_07_dual_code_integrity():
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
# 9. Meta-routing: global aggregation and dispute localization
# ---------------------------------------------------------------------------

def test_08_meta_routing(atlas_dir: Path):
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
# 10. Component isolation and rollback (BU-Ingress flavor)
# ---------------------------------------------------------------------------

def test_09_component_isolation_and_rollback():
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
    # Forward: s_curr = T_last_b(T_AA(s_prev))
    # Inverse: s_prev = T_AA^(-1)(T_last_b^(-1)(s_curr))
    # Since T_AA is involution: T_AA^(-1) = T_AA = R
    # And T_last_b^(-1) = R T_last_b R
    # So: s_prev = R (R T_last_b R)(s_curr) = R^2 T_last_b R(s_curr) = T_last_b R(s_curr)
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
# 11. Rollback tests (runtime features)
# ---------------------------------------------------------------------------

def test_10_shell_rollback(atlas_dir: Path):
    """
    Test shell rollback: create shells, rollback last one, verify totals revert.
    """
    print("\n=== SHELL ROLLBACK TEST ===")
    
    coord = Coordinator(atlas_dir)
    
    # Create two shells
    coord.add_grant("alice", UHI_PER_YEAR_MU * 3)
    coord.add_grant("bob", UHI_PER_YEAR_MU * 2)
    coord.close_shell(b"ecology:year:2026", CAPACITY_PER_YEAR_MU)
    
    coord.add_grant("alice", UHI_PER_YEAR_MU * 4)
    coord.add_grant("charlie", UHI_PER_YEAR_MU * 1)
    coord.close_shell(b"ecology:year:2027", CAPACITY_PER_YEAR_MU)
    
    # Capture totals after both shells
    status_before = coord.fiat_status()
    assert status_before["shell_count"] == 2
    assert status_before["used_capacity_MU"] == (UHI_PER_YEAR_MU * 3 + UHI_PER_YEAR_MU * 2 + UHI_PER_YEAR_MU * 4 + UHI_PER_YEAR_MU * 1)
    
    # Rollback last shell
    coord.rollback_last_shell()
    
    # Verify totals reverted
    status_after = coord.fiat_status()
    assert status_after["shell_count"] == 1
    assert status_after["used_capacity_MU"] == (UHI_PER_YEAR_MU * 3 + UHI_PER_YEAR_MU * 2)
    assert status_after["per_identity_totals"]["alice"] == UHI_PER_YEAR_MU * 3
    assert "charlie" not in status_after["per_identity_totals"]
    
    print("OK: shell rollback reverts totals correctly")


def test_11_kernel_rollback(atlas_dir: Path):
    """
    Test kernel rollback: step bytes, rollback, verify return to archetype.
    """
    print("\n=== KERNEL ROLLBACK TEST ===")
    
    coord = Coordinator(atlas_dir)
    
    # Step a known payload
    payload = b"test payload for rollback"
    coord.step_bytes(payload)
    
    assert coord.kernel.step == len(payload)
    assert len(coord.byte_log) == len(payload)
    
    # Rollback all bytes
    coord.rollback_kernel_steps(len(payload))
    
    # Verify returned to archetype
    sig_after = coord.kernel.signature()
    assert coord.kernel.step == 0
    assert len(coord.byte_log) == 0
    assert sig_after.state_index == coord.kernel.archetype_index
    
    print("OK: kernel rollback returns to archetype and clears byte_log")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", "-v", __file__]))