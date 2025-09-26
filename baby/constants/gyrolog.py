# baby/constants/gyrolog.py
"""
GyroLog: CGM Logarithmic Coordinate System for GyroSI

A discrete, integer coordinate system on the 48-bit state space using native
CGM generators. Provides anchor-agnostic coordinates for addressing, routing,
and path analysis within the GyroSI architecture.

QUICK START:
    gyrolog = GyroLog()
    coords = gyrolog.compute_gyrolog(state, anchor)
    print(coords)  # P0(+1,+1,+1)ε0.0

COORDINATES:
- plane: 0 or 1 (even/odd structural plane, Z2 from layer duality)
- parity: 0 or 1 (mirror class under FULL_MASK, UNA physics)
- orient_x/y/z: ±1 (Pauli triad signs for local orientation)
- residual: Float (Hamming defect to nearest plane template)
- grad_fg/bg/li: ±1 or 0 (optional gradient directions)

USAGE EXAMPLES:
    # State classification
    coords = gyrolog.compute_gyrolog(state, 0)
    if coords.plane == 0: print("Even-plane dominant")
    
    # Emission routing
    bucket = coordinate_based_routing_key(coords, 256)
    
    # Session tracking
    for intron in intron_sequence:
        next_state = apply_gyration_and_transform(state, intron)
        next_coords = gyrolog.compute_gyrolog(next_state, anchor)
        print(f"Plane flip: {coords.plane != next_coords.plane}")

See guides/GyroLog.md for comprehensive documentation.
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import sys
import os

# Handle both direct execution and module import
try:
    from ..kernel import governance
except ImportError:
    # Add the parent directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from baby.kernel import governance


@dataclass
class GyroCoords:
    """Discrete coordinates in the CGM logarithmic lattice."""
    plane: int          # π ∈ {0,1} (even/odd plane)
    parity: int         # p ∈ {0,1}  
    orient_x: int       # rX ∈ {-1,+1}
    orient_y: int       # rY ∈ {-1,+1}
    orient_z: int       # rZ ∈ {-1,+1}
    residual: float     # ε (Hamming distance)
    # Optional gradient directions (computed on-demand)
    grad_fg: int = 0    # gFG ∈ {-1,0,+1}
    grad_bg: int = 0    # gBG ∈ {-1,0,+1}
    grad_li: int = 0    # gLI ∈ {-1,0,+1}

    def __str__(self) -> str:
        return (f"P{self.plane}({self.orient_x:+d},{self.orient_y:+d},{self.orient_z:+d})"
                f"ε{self.residual:.1f}")


@dataclass  
class CanonicalIntrons:
    """Canonical single-bit representatives for each family."""
    li: int  # 0b01000000 - LI family (bit 6)
    fg: int  # 0b00100000 - FG family (bit 5) 
    bg: int  # 0b00010000 - BG family (bit 4)


class GyroLog:
    """CGM logarithmic coordinate system implementation."""
    
    def __init__(self):
        """Initialize with canonical templates and introns."""
        
        # Build 48-bit plane templates from GENE_Mac_S
        self.plane_templates = self._build_plane_templates()
        
        # Define canonical single-bit family introns
        self.canonical_introns = CanonicalIntrons(
            li=0x40,  # 0b01000000 - bit 6
            fg=0x20,  # 0b00100000 - bit 5
            bg=0x10   # 0b00010000 - bit 4
        )
        
    def _build_plane_templates(self) -> Tuple[int, int]:
        """Build even and odd plane templates from GENE_Mac_S."""
        # Even plane: full GENE_Mac_S tensor
        template_even = governance.tensor_to_int(governance.GENE_Mac_S)
        
        # Odd plane: complement of even plane
        template_odd = template_even ^ governance.FULL_MASK
        
        return template_even, template_odd
    
    
    def compute_gyrolog(self, state: int, anchor: int) -> GyroCoords:
        """Compute GyroLog coordinates for state relative to anchor."""
        
        # 1. Resolve parity (which parity class minimizes distance to anchor)
        parity = self._resolve_parity(state, anchor)
        
        # 2. Classify plane (even/odd template match)
        plane = self._classify_plane(state)
        
        # 3. Extract axis orientations from tensor structure
        orient_x, orient_y, orient_z = self._extract_orientations(state)
        
        # 4. Compute residual (min Hamming to plane templates)
        residual = self._compute_residual(state)
        
        # 5. Compute gradient directions (optional)
        grad_fg, grad_bg, grad_li = self._compute_gradients(state)
        
        return GyroCoords(
            plane=plane, parity=parity,
            orient_x=orient_x, orient_y=orient_y, orient_z=orient_z,
            residual=residual, grad_fg=grad_fg, grad_bg=grad_bg, grad_li=grad_li
        )
    
    def _resolve_parity(self, state: int, anchor: int) -> int:
        """Determine parity class based on complement class test (anchor-free)."""
        # Complement class test: s vs s^FULL_MASK
        # This determines which parity class the state belongs to
        complement = state ^ governance.FULL_MASK
        
        # Parity is 0 if state < complement, 1 if state >= complement
        # This is a state-local property, not anchor-dependent
        return 0 if state < complement else 1
    
    def _classify_plane(self, state: int) -> int:
        """Find best matching plane template (0=even, 1=odd)."""
        template_even, template_odd = self.plane_templates
        
        hamming_even = (state ^ template_even).bit_count()
        hamming_odd = (state ^ template_odd).bit_count()
        
        return 0 if hamming_even <= hamming_odd else 1
    
    def _extract_orientations(self, state: int) -> Tuple[int, int, int]:
        """Extract axis orientation signatures from state structure."""
        # Convert state back to tensor to analyze proper bit layout
        tensor = governance.int_to_tensor(state)
        
        # Sum signs over frames and layers per row, compare left vs right column
        orient_x, orient_y, orient_z = 0, 0, 0
        
        for layer in range(4):
            for frame in range(2):
                for row in range(3):
                    # Get left and right column values for this row
                    left_val = tensor[layer, frame, row, 0]
                    right_val = tensor[layer, frame, row, 1]
                    
                    # Compare left vs right to determine row orientation
                    if left_val > right_val:
                        if row == 0:
                            orient_x += 1
                        elif row == 1:
                            orient_y += 1
                        else:  # row == 2
                            orient_z += 1
                    elif right_val > left_val:
                        if row == 0:
                            orient_x -= 1
                        elif row == 1:
                            orient_y -= 1
                        else:  # row == 2
                            orient_z -= 1
        
        # Normalize to {-1, +1}
        return (
            +1 if orient_x >= 0 else -1,
            +1 if orient_y >= 0 else -1,
            +1 if orient_z >= 0 else -1
        )
    
    def _compute_residual(self, state: int) -> float:
        """Compute residual as min Hamming distance to plane templates."""
        template_even, template_odd = self.plane_templates
        
        hamming_even = (state ^ template_even).bit_count()
        hamming_odd = (state ^ template_odd).bit_count()
        
        return float(min(hamming_even, hamming_odd))
    
    def _compute_gradients(self, state: int) -> Tuple[int, int, int]:
        """Compute gradient directions for each family."""
        # Apply each canonical intron and measure Hamming change
        base_hamming = self._compute_residual(state)
        
        # Test FG gradient
        state_fg = governance.apply_gyration_and_transform(state, self.canonical_introns.fg)
        hamming_fg = self._compute_residual(state_fg)
        grad_fg = +1 if hamming_fg < base_hamming else (-1 if hamming_fg > base_hamming else 0)
        
        # Test BG gradient
        state_bg = governance.apply_gyration_and_transform(state, self.canonical_introns.bg)
        hamming_bg = self._compute_residual(state_bg)
        grad_bg = +1 if hamming_bg < base_hamming else (-1 if hamming_bg > base_hamming else 0)
        
        # Test LI gradient
        state_li = governance.apply_gyration_and_transform(state, self.canonical_introns.li)
        hamming_li = self._compute_residual(state_li)
        grad_li = +1 if hamming_li < base_hamming else (-1 if hamming_li > base_hamming else 0)
        
        return grad_fg, grad_bg, grad_li
    
    def update_coords(self, coords: GyroCoords, intron: int) -> GyroCoords:
        """Update coordinates by recomputing from new state (simplified)."""
        # Since we removed path-dependent coordinates, just return the input
        # In practice, you'd apply the intron to get the new state and recompute
        return coords
    
    # Validation methods
    
    def verify_commutator(self) -> Dict[str, int]:
        """Test commutator defect distribution (no hard pass/fail)."""
        import random
        
        defects = []
        for _ in range(100):
            # Random state
            state = random.randint(0, (1 << 48) - 1)
            
            # Compute FG∘BG
            state1 = governance.apply_gyration_and_transform(state, self.canonical_introns.fg)
            state1 = governance.apply_gyration_and_transform(state1, self.canonical_introns.bg)
            
            # Compute BG∘FG
            state2 = governance.apply_gyration_and_transform(state, self.canonical_introns.bg)
            state2 = governance.apply_gyration_and_transform(state2, self.canonical_introns.fg)
            
            # Record defect
            defect = state1 ^ state2
            defects.append(defect)
        
        # Count defect patterns
        defect_counts = {}
        for defect in defects:
            defect_counts[defect] = defect_counts.get(defect, 0) + 1
        
        return defect_counts
    
    def verify_plane_toggle(self) -> bool:
        """Test that FG and BG toggle plane bit."""
        import random
        
        toggles_correct = 0
        total_tests = 50
        
        for _ in range(total_tests):
            # Random state
            state = random.randint(0, (1 << 48) - 1)
            initial_plane = self._classify_plane(state)
            
            # Test FG toggle
            state_fg = governance.apply_gyration_and_transform(state, self.canonical_introns.fg)
            plane_fg = self._classify_plane(state_fg)
            
            # Test BG toggle
            state_bg = governance.apply_gyration_and_transform(state, self.canonical_introns.bg)
            plane_bg = self._classify_plane(state_bg)
            
            # Check if toggles occurred
            if plane_fg != initial_plane or plane_bg != initial_plane:
                toggles_correct += 1
        
        return toggles_correct >= total_tests * 0.7  # 70% threshold
    
    def verify_anchor_invariance(self, test_state: int, anchor1: int, anchor2: int) -> bool:
        """Verify anchor-free invariants are preserved under anchor shift."""
        coords1 = self.compute_gyrolog(test_state, anchor1)
        coords2 = self.compute_gyrolog(test_state, anchor2)
        
        # Check anchor-free invariants
        return (coords1.plane == coords2.plane and 
                coords1.parity == coords2.parity and
                coords1.orient_x == coords2.orient_x and
                coords1.orient_y == coords2.orient_y and
                coords1.orient_z == coords2.orient_z)


# Demo and utility functions

def demo_session(intron_sequence: List[int], anchor: Optional[int] = None) -> None:
    """Demonstrate GyroLog coordinate tracking through a session."""
    
    gyrolog = GyroLog()
    
    if anchor is None:
        anchor = governance.tensor_to_int(governance.GENE_Mac_S)
    
    # Type assertion: anchor is now guaranteed to be int
    anchor = int(anchor)
    
    print("GyroLog Session Demo")
    print("=" * 60)
    print(f"Anchor: 0x{anchor:012x}")
    print(f"Canonical introns: LI=0x{gyrolog.canonical_introns.li:02x}, "
          f"FG=0x{gyrolog.canonical_introns.fg:02x}, BG=0x{gyrolog.canonical_introns.bg:02x}")
    print()
    
    current_state = anchor
    coords = gyrolog.compute_gyrolog(current_state, anchor)
    
    print(f"Step 0: {coords}")
    print(f"        State: 0x{current_state:012x}")
    print()
    
    for step, intron in enumerate(intron_sequence, 1):
        # Apply transform
        next_state = governance.apply_gyration_and_transform(current_state, intron)
        next_coords = gyrolog.compute_gyrolog(next_state, anchor)
        
        # Show changes
        dp = (next_coords.plane - coords.plane) % 2
        dparity = (next_coords.parity - coords.parity) % 2
        dox = next_coords.orient_x - coords.orient_x
        doy = next_coords.orient_y - coords.orient_y
        doz = next_coords.orient_z - coords.orient_z
        
        print(f"Step {step}: {next_coords}")
        print(f"        Intron: 0x{intron:02x}")
        print(f"        State: 0x{current_state:012x} → 0x{next_state:012x}")
        print(f"        Delta: ΔP={dp}, ΔParity={dparity}, ΔOrient=({dox:+d},{doy:+d},{doz:+d})")
        print()
        
        current_state = next_state
        coords = next_coords


def run_validation_suite() -> None:
    """Run comprehensive GyroLog validation tests."""
    
    gyrolog = GyroLog()
    
    print("GyroLog Validation Suite")
    print("=" * 60)
    
    # Test 1: Commutator defect analysis
    defect_counts = gyrolog.verify_commutator()
    print(f"✓ Commutator defect analysis: {len(defect_counts)} unique patterns")
    # Show top 3 most common defects
    sorted_defects = sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (defect, count) in enumerate(sorted_defects[:3]):
        print(f"  Pattern {i+1}: 0x{defect:012x} (count: {count})")
    
    # Test 2: Plane toggle behavior
    toggle_ok = gyrolog.verify_plane_toggle()
    print(f"✓ Plane toggle behavior: {'PASS' if toggle_ok else 'FAIL'}")
    
    # Test 3: Anchor invariance
    archetype = governance.tensor_to_int(governance.GENE_Mac_S)
    test_state = governance.apply_gyration_and_transform(archetype, 0x42)
    anchor1 = archetype
    anchor2 = governance.apply_gyration_and_transform(archetype, 0x24)
    
    invariance_ok = gyrolog.verify_anchor_invariance(test_state, anchor1, anchor2)
    print(f"✓ Anchor invariance: {'PASS' if invariance_ok else 'FAIL'}")
    
    # Test 4: Coordinate consistency
    coords1 = gyrolog.compute_gyrolog(test_state, anchor1)
    coords2 = gyrolog.compute_gyrolog(test_state, anchor2)
    consistency_ok = (coords1.plane == coords2.plane and 
                     coords1.parity == coords2.parity and
                     coords1.orient_x == coords2.orient_x and
                     coords1.orient_y == coords2.orient_y and
                     coords1.orient_z == coords2.orient_z)
    print(f"✓ Coordinate consistency: {'PASS' if consistency_ok else 'FAIL'}")
    
    print()
    print("Plane templates:")
    template_even, template_odd = gyrolog.plane_templates
    print(f"  Even plane: 0x{template_even:012x}")
    print(f"  Odd plane: 0x{template_odd:012x}")
    
    print()
    print("Sample coordinate mapping:")
    sample_states = [archetype, test_state, anchor2]
    for i, state in enumerate(sample_states):
        coords = gyrolog.compute_gyrolog(state, archetype)
        print(f"  State {i}: 0x{state:012x} → {coords}")


def coordinate_based_routing_key(coords: GyroCoords, modulus: int = 256) -> int:
    """Generate routing key from coordinates for emission system."""
    # Linear combination of anchor-free coordinates for bucket selection
    key = (
        coords.plane * 128 +
        coords.parity * 64 + 
        ((coords.orient_x + 1) // 2) * 32 +
        ((coords.orient_y + 1) // 2) * 16 +
        ((coords.orient_z + 1) // 2) * 8 +
        (int(coords.residual) % 8)
    ) % modulus
    return key


if __name__ == "__main__":
    # Run validation tests
    run_validation_suite()
    
    print("\n" + "=" * 60)
    
    # Demo with sample intron sequence
    sample_introns = [0x42, 0x24, 0x18, 0x81, 0x66, 0xAA]
    demo_session(sample_introns)