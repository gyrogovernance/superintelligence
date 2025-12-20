# GyroLog: CGM Logarithmic Coordinate System

## Overview

GyroLog is a **physics-grounded coordinate system** for GyroSI states that maps 48-bit integers into meaningful geometric coordinates based on CGM (Common Governance Model) principles. It's not just a benchmark tool—it's a practical system for state navigation, routing, debugging, and physics simulation within the GyroSI framework.

## What GyroLog Actually Is

GyroLog provides "GPS coordinates" for points on your finite manifold by converting 48-bit GyroSI states into a small set of integers that capture their geometric structure according to CGM physics.

### Input/Output
- **Input**: A 48-bit state integer (and optionally an anchor state for relative positioning)
- **Output**: A `GyroCoords` object with:
  - `plane`: 0 or 1 (even/odd structural plane, modeling Z2 from layer duality)
  - `parity`: 0 or 1 (mirror class under FULL_MASK, directly from UNA physics)
  - `orient_x/y/z`: ±1 (axis signs, Pauli triad rX,rY,rZ for local orientation)
  - `residual`: Float (Hamming defect to nearest plane template—measures "noise" relative to canonical CGM forms)
  - `grad_fg/bg/li`: ±1 or 0 (optional "directions" showing how applying FG/BG/LI would change the residual)

## Physics Foundation

GyroLog is derived from core CGM components:
- **GENE_Mac_S tensor**: Templates for plane classification
- **Family masks**: LI/FG/BG for gradient computation
- **FULL_MASK**: For parity determination
- **apply_gyration_and_transform**: For gradient computation

The coordinates are **anchor-agnostic** in their invariants (plane, parity, orientations, gradients)—these depend only on the state itself, not the anchor.

## Core Usage

### Basic Coordinate Computation

```python
from baby.constants.gyrolog import GyroLog

# Initialize the coordinate system
gyrolog = GyroLog()

# Compute coordinates for a state
state = 0xa9556aa9556a  # Example 48-bit state
anchor = 0xa9556aa9556a  # Reference state (optional)

coords = gyrolog.compute_gyrolog(state, anchor)
print(coords)  # P0(+1,+1,+1)ε0.0

# Access individual coordinates
print(f"Plane: {coords.plane}")        # 0 or 1
print(f"Parity: {coords.parity}")      # 0 or 1  
print(f"Orientations: ({coords.orient_x}, {coords.orient_y}, {coords.orient_z})")
print(f"Residual: {coords.residual}")  # Float
print(f"Gradients: FG={coords.grad_fg}, BG={coords.grad_bg}, LI={coords.grad_li}")
```

### State Classification

```python
# Quickly classify states
def classify_state(state):
    coords = gyrolog.compute_gyrolog(state, 0)  # Anchor doesn't matter for invariants
    
    if coords.plane == 0:
        plane_type = "even-plane dominant (CS/UNA-like)"
    else:
        plane_type = "odd-plane dominant (ONA/BU-like)"
    
    if coords.parity == 0:
        parity_type = "low parity (canonical)"
    else:
        parity_type = "high parity (mirror-flipped)"
    
    return f"{plane_type}, {parity_type}, residual={coords.residual:.1f}"

# Example usage
state1 = 0xa9556aa9556a  # Archetype
state2 = 0x56aa9556aa95  # Complement
print(classify_state(state1))  # even-plane dominant, low parity, residual=0.0
print(classify_state(state2))  # odd-plane dominant, high parity, residual=0.0
```

## Practical Applications

### 1. Emission Routing

Use coordinates for bucket selection in phase-propagating emission:

```python
from baby.constants.gyrolog import coordinate_based_routing_key

def route_emission(state, num_buckets=256):
    coords = gyrolog.compute_gyrolog(state, 0)
    bucket = coordinate_based_routing_key(coords, num_buckets)
    return bucket

# Example: Route states to different buckets based on their structure
states = [0xa9556aa9556a, 0x14a89514a895, 0x895a91895a91]
for state in states:
    bucket = route_emission(state)
    print(f"State 0x{state:012x} → Bucket {bucket}")
```

### 2. Session Tracking

Monitor coordinate changes during a session to track helical progression:

```python
def track_session(intron_sequence, initial_state):
    """Track coordinate changes through a session."""
    gyrolog = GyroLog()
    current_state = initial_state
    coords = gyrolog.compute_gyrolog(current_state, initial_state)
    
    print(f"Initial: {coords}")
    
    for step, intron in enumerate(intron_sequence, 1):
        # Apply transform
        next_state = governance.apply_gyration_and_transform(current_state, intron)
        next_coords = gyrolog.compute_gyrolog(next_state, initial_state)
        
        # Show changes
        plane_flip = "YES" if next_coords.plane != coords.plane else "NO"
        parity_flip = "YES" if next_coords.parity != coords.parity else "NO"
        
        print(f"Step {step}: {next_coords}")
        print(f"  Plane flip: {plane_flip}, Parity flip: {parity_flip}")
        print(f"  Intron: 0x{intron:02x}")
        
        current_state = next_state
        coords = next_coords

# Example usage
introns = [0x42, 0x24, 0x18, 0x81, 0x66, 0xAA]
track_session(introns, 0xa9556aa9556a)
```

### 3. Physics Simulation

Simulate CGM stage progression and measure physical properties:

```python
import random

def simulate_cgm_path(start_state, steps, valid_introns):
    """Simulate CGM progression and track coordinates."""
    gyrolog = GyroLog()
    path = []
    current_state = start_state
    
    for step in range(steps):
        coords = gyrolog.compute_gyrolog(current_state, start_state)
        path.append((step, current_state, coords))
        
        # Apply random valid intron
        intron = random.choice(valid_introns)
        current_state = governance.apply_gyration_and_transform(current_state, intron)
    
    return path

def analyze_physics_properties(path):
    """Analyze physical properties of the path."""
    plane_flips = 0
    parity_flips = 0
    total_residual = 0
    
    for i in range(1, len(path)):
        prev_coords = path[i-1][2]
        curr_coords = path[i][2]
        
        if prev_coords.plane != curr_coords.plane:
            plane_flips += 1
        if prev_coords.parity != curr_coords.parity:
            parity_flips += 1
        
        total_residual += curr_coords.residual
    
    avg_residual = total_residual / len(path)
    
    print(f"Physics Analysis:")
    print(f"  Plane flips: {plane_flips} ({plane_flips/len(path)*100:.1f}%)")
    print(f"  Parity flips: {parity_flips} ({parity_flips/len(path)*100:.1f}%)")
    print(f"  Average residual: {avg_residual:.2f}")

# Example usage
valid_introns = [0x40, 0x20, 0x10, 0x42, 0x24, 0x18, 0x81, 0x66, 0xAA]
path = simulate_cgm_path(0xa9556aa9556a, 100, valid_introns)
analyze_physics_properties(path)
```

### 4. Debugging and Validation

Use coordinates to debug unexpected behavior:

```python
def debug_transformation(state, intron):
    """Debug a transformation by analyzing coordinate changes."""
    gyrolog = GyroLog()
    
    coords_before = gyrolog.compute_gyrolog(state, 0)
    next_state = governance.apply_gyration_and_transform(state, intron)
    coords_after = gyrolog.compute_gyrolog(next_state, 0)
    
    print(f"Debug Transformation:")
    print(f"  Before: {coords_before}")
    print(f"  After:  {coords_after}")
    print(f"  Intron: 0x{intron:02x}")
    
    # Check for expected behaviors
    if intron & 0x20:  # FG intron
        expected_plane_flip = "YES"
        actual_plane_flip = "YES" if coords_before.plane != coords_after.plane else "NO"
        print(f"  FG plane toggle: Expected {expected_plane_flip}, Got {actual_plane_flip}")
    
    if intron & 0x10:  # BG intron
        expected_plane_flip = "YES"
        actual_plane_flip = "YES" if coords_before.plane != coords_after.plane else "NO"
        print(f"  BG plane toggle: Expected {expected_plane_flip}, Got {actual_plane_flip}")

# Example usage
debug_transformation(0xa9556aa9556a, 0x20)  # FG intron
```

## Coordinate Interpretation

### Plane (0 or 1)
- **0 (Even)**: CS/UNA-like structural plane, closer to archetype template
- **1 (Odd)**: ONA/BU-like structural plane, closer to complement template
- **Physics**: Models Z2 from layer duality in CGM

### Parity (0 or 1)
- **0**: Low parity (state < complement)
- **1**: High parity (state ≥ complement)
- **Physics**: Mirror class under FULL_MASK, directly from UNA physics

### Orientations (±1 each)
- **orient_x/y/z**: Pauli triad signs for local axis alignment
- **Physics**: Discrete Pauli signs for SU(2) rotations and chirality

### Residual (Float)
- **Low (0-6)**: Close to canonical CGM form (BU-aligned)
- **High (20+)**: Differentiated state (ONA-like)
- **Physics**: Like gyrotriangle defect δ in CGM

### Gradients (±1 or 0)
- **grad_fg/bg/li**: Direction showing how applying that family would change residual
- **+1**: Applying intron would improve (lower residual)
- **-1**: Applying intron would worsen (higher residual)
- **0**: No change in residual

## Validation and Testing

The validation suite ensures GyroLog correctly models CGM physics:

```python
from baby.constants.gyrolog import run_validation_suite

# Run all validation tests
run_validation_suite()
```

### Test Results Interpretation

- **Commutator Defect Analysis**: Shows consistent defect patterns (not failures)
- **Plane Toggle Behavior**: Validates Z2 plane responsiveness
- **Anchor Invariance**: Confirms anchor-free invariants
- **Coordinate Consistency**: Ensures stable computation

## Integration with GyroSI

### Memory Storage
```python
def store_with_coords(state, memory_system):
    """Store state with its coordinates for efficient retrieval."""
    coords = gyrolog.compute_gyrolog(state, 0)
    
    # Use coordinates for indexing
    bucket = coordinate_based_routing_key(coords)
    
    # Store with coordinate metadata
    memory_system.store(state, bucket, {
        'plane': coords.plane,
        'parity': coords.parity,
        'residual': coords.residual
    })
```

### Emission Integration
```python
def coordinate_guided_emission(current_state, target_coords):
    """Use coordinates to guide emission toward target structure."""
    gyrolog = GyroLog()
    current_coords = gyrolog.compute_gyrolog(current_state, 0)
    
    # Find introns that move toward target
    suitable_introns = []
    for intron in valid_introns:
        test_state = governance.apply_gyration_and_transform(current_state, intron)
        test_coords = gyrolog.compute_gyrolog(test_state, 0)
        
        # Check if this intron improves alignment
        if test_coords.plane == target_coords.plane:
            suitable_introns.append(intron)
    
    return suitable_introns
```

## Advanced Usage

### Custom Coordinate Analysis
```python
def analyze_coordinate_distribution(states):
    """Analyze distribution of coordinates across a set of states."""
    gyrolog = GyroLog()
    
    planes = {'even': 0, 'odd': 0}
    parities = {'low': 0, 'high': 0}
    residuals = []
    
    for state in states:
        coords = gyrolog.compute_gyrolog(state, 0)
        
        planes['even' if coords.plane == 0 else 'odd'] += 1
        parities['low' if coords.parity == 0 else 'high'] += 1
        residuals.append(coords.residual)
    
    print(f"Plane distribution: {planes}")
    print(f"Parity distribution: {parities}")
    print(f"Residual stats: min={min(residuals):.1f}, max={max(residuals):.1f}, avg={sum(residuals)/len(residuals):.1f}")
```

### Physics Experimentation
```python
def measure_commutator_defects():
    """Measure and analyze commutator defects in the system."""
    gyrolog = GyroLog()
    defect_counts = gyrolog.verify_commutator()
    
    print("Commutator Defect Analysis:")
    for defect, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Pattern 0x{defect:012x}: {count} occurrences")
        
        # Analyze the defect pattern
        weight = bin(defect).count('1')
        print(f"    Hamming weight: {weight}")
        
        # Check if it relates to family masks
        if defect & 0x20:
            print(f"    Contains FG bits (0x20)")
        if defect & 0x10:
            print(f"    Contains BG bits (0x10)")
```

## Summary

GyroLog is a powerful physics tool that provides meaningful coordinates for GyroSI states based on CGM principles. It's designed for:

1. **State Classification**: Quickly identify structural properties
2. **Emission Routing**: Use coordinates for bucket selection
3. **Session Tracking**: Monitor coordinate changes over time
4. **Physics Simulation**: Model CGM progression and measure properties
5. **Debugging**: Analyze transformations and unexpected behavior

The validation tests ensure the math is correct, but the core value is in using the coordinate system for practical GyroSI operations and physics simulation.

---

*"GyroLog gives your states 'CGM-meaningful labels' (like coordinates on a map) that you can compute fast and use for practical things in GyroSI. The script's tests are just proof it's wired correctly to your physics engine."*
