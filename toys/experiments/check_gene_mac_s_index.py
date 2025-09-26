#!/usr/bin/env python3
"""
Check the actual index of GENE_Mac_S in the current ontology.
"""

import numpy as np
from pathlib import Path
from baby.governance import GENE_Mac_S
from baby.information import InformationEngine


def check_gene_mac_s_index():
    """Find the actual index of GENE_Mac_S in the current ontology."""

    # Load the corrected ontology
    meta_path = Path("memories/public/meta")
    ontology_path = meta_path / "ontology_keys.npy"

    if not ontology_path.exists():
        print("❌ Ontology file not found.")
        return

    ontology = np.load(ontology_path, mmap_mode="r")

    # Convert GENE_Mac_S to integer
    archetypal_int = InformationEngine.tensor_to_int(GENE_Mac_S)

    print(f"📊 GENE_Mac_S Analysis:")
    print(f"   Tensor shape: {GENE_Mac_S.shape}")
    print(f"   Integer value: {archetypal_int} (0x{archetypal_int:012X})")

    # Find its index in the ontology
    indices = np.where(ontology == archetypal_int)[0]

    if len(indices) > 0:
        index = indices[0]
        print(f"   Index in ontology: {index}")
        print(f"   Total states: {len(ontology)}")
        print(f"   Position: {index} of {len(ontology)}")

        # Load theta to see its theta value
        theta_path = meta_path / "theta.npy"
        if theta_path.exists():
            theta = np.load(theta_path, mmap_mode="r")
            theta_val = float(theta[index])
            print(f"   Theta value: {theta_val:.6f}")

            # Compare with key angles
            import math

            angles = {"0": 0, "π/4": math.pi / 4, "π/2": math.pi / 2, "3π/4": 3 * math.pi / 4, "π": math.pi}

            closest_angle = None
            min_diff = float("inf")

            for name, angle in angles.items():
                diff = abs(theta_val - angle)
                if diff < min_diff:
                    min_diff = diff
                    closest_angle = name

            print(f"   Closest angle: {closest_angle} (diff: {min_diff:.6f})")

        return {"archetypal_int": archetypal_int, "archetypal_index": index, "total_states": len(ontology)}
    else:
        print(f"   ❌ GENE_Mac_S not found in ontology!")
        print(f"   This indicates a serious problem with the ontology.")
        return None


if __name__ == "__main__":
    result = check_gene_mac_s_index()
    if result:
        print(f"\n✅ GENE_Mac_S found at index {result['archetypal_index']}")
    else:
        print(f"\n❌ GENE_Mac_S not found in ontology")
