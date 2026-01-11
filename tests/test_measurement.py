"""
Measurement tests: scalar aggregation, structural lock, and information loss.

These tests prove two fundamental limitations of empirical (entity-focused) evaluations:
1. SCALAR BLINDNESS: Aggregating 6D structure into 1D scalars loses the information 
   required to distinguish aligned states from misaligned ones.
2. STRUCTURAL LOCK: Optimizing for single-axis metrics (e.g., pure GTD) mathematically 
   locks the system to a fixed aperture (A=0.5), making it impossible to reach 
   the alignment target A* = 0.0207 regardless of effort.

Only epistemic (multi-axis) enumeration preserves the degrees of freedom required 
to calibrate the system to A*.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.app.cli.schemas import A_STAR
from src.app.events import Domain
from src.app.ledger import (
    DomainLedgers,
    compute_aperture,
    construct_edge_vector_with_aperture,
    get_projections,
    hodge_decomposition,
)
from src.plugins.frameworks import PluginContext, THMDisplacementPlugin


class TestMeasurementCollapse:
    """
    Rigorously demonstrate the failures of empirical evaluation modes via K4 geometry.
    """

    def _apply_thm_payload(self, ledgers: DomainLedgers, payload: Dict[str, Any], domain_str: str = "economy"):
        """
        Helper: run THMDisplacementPlugin on a payload and apply resulting events
        to the given DomainLedgers for the specified domain.
        """
        plugin = THMDisplacementPlugin()
        ctx = PluginContext(meta={"test": "measurement"})
        events = plugin.emit_events({**payload, "domain": domain_str}, ctx)
        for ev in events:
            ledgers.apply_event(ev)

    def test_scalar_collapse_loses_aperture_distinguishability(self):
        """
        Proof of Blindness 1:
        Two states with identical empirical scores (Scalar Sum = 4.0) can have 
        structurally distinct apertures. The scalar cannot distinguish them.
        """
        P_grad, P_cycle = get_projections()

        # State 1: Collapsed (Single Axis)
        y1 = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # State 2: Distributed
        y2 = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)

        # Compute apertures
        _, y1_cycle = hodge_decomposition(y1, P_grad, P_cycle)
        A1 = compute_aperture(y1, y1_cycle)

        _, y2_cycle = hodge_decomposition(y2, P_grad, P_cycle)
        A2 = compute_aperture(y2, y2_cycle)

        # Scalar sums (Empirical Score)
        scalar1 = float(np.sum(np.abs(y1)))
        scalar2 = float(np.sum(np.abs(y2)))

        print("\n" + "="*10)
        print("PROOF: SCALAR BLINDNESS")
        print("="*10)
        print(f"  State 1 (Collapsed):   Scalar = {scalar1:.6f}, A = {A1:.6f}")
        print(f"  State 2 (Distributed): Scalar = {scalar2:.6f}, A = {A2:.6f}")
        print(f"  Scalar distinguishes states? {abs(scalar1 - scalar2) > 1e-6}")
        print("  ✓ Proven: Scalar aggregation discards structural information")

        assert abs(scalar1 - scalar2) < 1e-6, "Scalar scores must be identical"
        assert abs(A1 - A2) > 1e-6, "Structural states must be different"

    def test_scalar_sum_cannot_detect_A_star_proximity(self):
        """
        Proof of Blindness 2:
        Scalar aggregation cannot tell if a system is Aligned (near A*) or 
        Misaligned (far from A*).
        """
        P_grad, P_cycle = get_projections()

        # Construct Aligned State (y_near)
        x_near = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        y_near = construct_edge_vector_with_aperture(x_near, target_aperture=A_STAR)
        _, y_near_cycle = hodge_decomposition(y_near, P_grad, P_cycle)
        A_near = compute_aperture(y_near, y_near_cycle)

        # Construct Misaligned State (y_far, A=0.5)
        x_far = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        y_far = construct_edge_vector_with_aperture(x_far, target_aperture=0.5)

        # Normalize y_far to have the same empirical score as y_near
        scalar_near = float(np.sum(np.abs(y_near)))
        scalar_far_orig = float(np.sum(np.abs(y_far)))
        y_far_scaled = y_far * (scalar_near / scalar_far_orig)
        
        # Recompute A_far (Aperture is scale invariant, so it stays 0.5)
        _, y_far_scaled_cycle = hodge_decomposition(y_far_scaled, P_grad, P_cycle)
        A_far_scaled = compute_aperture(y_far_scaled, y_far_scaled_cycle)
        
        scalar_far = float(np.sum(np.abs(y_far_scaled)))

        dist_near = abs(A_near - A_STAR)
        dist_far = abs(A_far_scaled - A_STAR)

        print("\n" + "="*10)
        print("PROOF: ALIGNMENT INVISIBILITY")
        print("="*10)
        print(f"  Aligned State:    Scalar = {scalar_near:.6f}, |A - A*| = {dist_near:.6f}")
        print(f"  Misaligned State: Scalar = {scalar_far:.6f}, |A - A*| = {dist_far:.6f}")
        print("  ✓ Proven: Scalar evaluation cannot detect alignment")

        assert abs(scalar_near - scalar_far) < 1e-6
        assert dist_near < 1e-6
        assert dist_far > 0.4  # Massive misalignment

    def test_single_axis_structural_lock_vs_multi_axis_freedom(self):
        """
        Proof of Structural Lock:
        1. Single-Axis: Demonstrate that pushing ANY single THM axis (e.g., GTD) 
           locks the system to A=0.5. It is mathematically impossible to reach A* = 0.0207.
        2. Multi-Axis Freedom: Demonstrate that Epistemic enumeration allows constructing 
           a vector that hits A* exactly.
        """
        P_grad, P_cycle = get_projections()

        # --- PART 1: The Trap (Single Axis Lock) ---
        # Try to align using only GTD (Governance-Information)
        # We can increase magnitude, but we cannot change structure.
        magnitudes = [1.0, 10.0, 100.0]
        single_axis_apertures = []
        
        for mag in magnitudes:
            ledgers = DomainLedgers()
            self._apply_thm_payload(ledgers, {"GTD": mag}, domain_str="education")
            A = ledgers.aperture(Domain.EDUCATION)
            single_axis_apertures.append(A)

        # --- PART 2: The Solution (Multi-Axis Freedom) ---
        # Construct a vector that hits A* exactly
        x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        y_aligned = construct_edge_vector_with_aperture(x, target_aperture=A_STAR)
        _, y_cyc = hodge_decomposition(y_aligned, P_grad, P_cycle)
        A_aligned = compute_aperture(y_aligned, y_cyc)

        print("\n" + "="*10)
        print("PROOF: STRUCTURAL LOCK VS FREEDOM")
        print("="*10)
        print(f"  Target A* = {A_STAR:.6f}")
        print("  [Single Axis Strategy]")
        for i, m in enumerate(magnitudes):
            print(f"    Mag {m:5.1f} -> A = {single_axis_apertures[i]:.6f} (Locked)")
        
        print("  [Multi-Axis Strategy]")
        print(f"    Constructed y -> A = {A_aligned:.6f} (Aligned)")
        
        print("  ✓ Proven: Single-axis optimization cannot achieve alignment.")
        print("  ✓ Proven: Epistemic enumeration enables A*.")

        # Assert Lock
        for A in single_axis_apertures:
            assert abs(A - 0.5) < 1e-6, "Single axis must be locked to 0.5"
            assert abs(A - A_STAR) > 0.4, "Single axis cannot reach A*"
            
        # Assert Freedom
        assert abs(A_aligned - A_STAR) < 1e-6, "Multi-axis must hit A*"

    def test_kernel_A_kernel_vs_app_A_star(self):
        """
        Bridge the kernel-level discrete aperture (A_kernel = 5/256) with the
        App-level continuous aperture (A* = 0.0207).
        """
        A_kernel = 5.0 / 256.0
        rel_diff = abs(A_kernel - A_STAR) / A_STAR * 100.0

        print("\n" + "="*10)
        print("KERNEL ↔ APP APERTURE BRIDGE")
        print("="*10)
        print(f"  A_kernel (discrete) = {A_kernel:.12f} (5/256)")
        print(f"  A* (CGM continuous) = {A_STAR:.12f}")
        print(f"  Relative difference = {rel_diff:.1f}%")
        print("  ✓ Kernel discrete structure approximates CGM continuous target")

        assert rel_diff < 6.0

    def test_epistemic_aperture_is_scale_invariant(self):
        """
        Prove Self-Normalization:
        Epistemic measurement (aperture) is invariant under global scaling.
        Scalar measurement is not.
        """
        P_grad, P_cycle = get_projections()
        
        # Base vector (structurally rich)
        y = np.array([1.0, 2.0, 0.5, 1.5, 0.2, 1.0], dtype=np.float64)
        _, y_cycle = hodge_decomposition(y, P_grad, P_cycle)
        A_base = compute_aperture(y, y_cycle)
        S_base = float(np.sum(np.abs(y)))

        # Scaled vector
        lam = 100.0
        y_scaled = y * lam
        _, ys_cycle = hodge_decomposition(y_scaled, P_grad, P_cycle)
        A_scaled = compute_aperture(y_scaled, ys_cycle)
        S_scaled = float(np.sum(np.abs(y_scaled)))

        print("\n" + "="*10)
        print("PROOF: EPISTEMIC SELF-NORMALIZATION")
        print("="*10)
        print(f"  Scalar: {S_base:.4f} -> {S_scaled:.4f} (Scales with input)")
        print(f"  Aperture: {A_base:.12f} -> {A_scaled:.12f} (Invariant)")
        print("  ✓ Epistemic measurement is self-normalizing")

        assert abs(A_base - A_scaled) < 1e-12
        assert abs(S_scaled - (S_base * lam)) < 1e-6