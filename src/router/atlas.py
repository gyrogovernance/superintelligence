"""
Atlas builder.

Builds the three compiled artifacts of the kernel physics:

    python -m src.router.atlas              # default output: data/atlas/
    python -m src.router.atlas --out path/  # custom output directory

Artifacts produced:
    ontology.npy        Sorted unique reachable 24-bit states (65,536 entries)
    epistemology.npy    [N, 256] next-state index lookup table
    phenomenology.npz   Spectral atlas (see below)

Phenomenology contains six categories of compiled knowledge:

    1. Kernel constants       Macro archetype state (ARCHETYPE_STATE24), GENE_MIC_S,
                              masks, dual code vectors
    2. Intron-stage priors    Byte structure BEFORE mask expansion (Appendix G):
                              micro-ref, family, CGM stage parities, features
    3. Per-state observables  state_horizon[N], state_vertex[N]
    4. Spectral phase cube    phase[N, 256]
    5. Backward-pass maps     next_horizon[N,256], next_vertex[N,256],
                              next_phase[N,256]
    6. Mask-stage helpers     mask12_by_byte[256], byte_weight, byte_charge,
                              features_K*

Categories 2 and 6 both describe bytes from different stages:
    2 = BEFORE projection (intron space, constitutional priors preserved)
    6 = AFTER projection (mask space, geometry-compatible with 24-bit state)

Rebuild required when constants.py changes or atlas version increments.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Import constants - handle both direct execution and module import
try:
    from .constants import (
        ARCHETYPE_A12,
        ARCHETYPE_B12,
        ARCHETYPE_STATE24,
        C_PERP_12,
        GENE_MIC_S,
        LAYER_MASK_12,
        Q0,
        Q1,
        XFORM_MASK_BY_BYTE,
        mask12_for_byte,
        popcount,
        vertex_charge_from_mask,
    )
except ImportError:
    program_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(program_root))

    from src.router.constants import (
        ARCHETYPE_A12,
        ARCHETYPE_B12,
        ARCHETYPE_STATE24,
        C_PERP_12,
        GENE_MIC_S,
        LAYER_MASK_12,
        Q0,
        Q1,
        XFORM_MASK_BY_BYTE,
        mask12_for_byte,
        popcount,
        vertex_charge_from_mask,
    )


@dataclass(frozen=True)
class AtlasPaths:
    base: Path

    @property
    def ontology(self) -> Path:
        return self.base / "ontology.npy"

    @property
    def epistemology(self) -> Path:
        return self.base / "epistemology.npy"

    @property
    def phenomenology(self) -> Path:
        return self.base / "phenomenology.npz"


# Atlas version - increment when build method or constants change incompatibly
ATLAS_VERSION = "2.2"


def build_ontology(paths: AtlasPaths) -> NDArray[np.uint32]:
    """
    Build ontology directly as A_set x B_set using proven closed-form algebra.

    By Property P3, Omega = A_set x B_set where:
    - A_set = {ARCHETYPE_A12 XOR m_b : b in [0,255]} (256 elements)
    - B_set = {ARCHETYPE_B12 XOR m_b : b in [0,255]} (256 elements)
    """
    print("Building ontology...")
    paths.base.mkdir(parents=True, exist_ok=True)

    masks_a = np.array(
        [(int(XFORM_MASK_BY_BYTE[b]) >> 12) & 0xFFF for b in range(256)],
        dtype=np.uint16,
    )

    a_set = (ARCHETYPE_A12 ^ masks_a).astype(np.uint16) & LAYER_MASK_12
    b_set = (ARCHETYPE_B12 ^ masks_a).astype(np.uint16) & LAYER_MASK_12

    a_grid, b_grid = np.meshgrid(a_set, b_set, indexing="ij")

    ontology = ((a_grid.astype(np.uint32) << 12) | b_grid.astype(np.uint32)).flatten()
    ontology = np.sort(ontology).astype(np.uint32)

    np.save(paths.ontology, ontology)

    file_size = paths.ontology.stat().st_size
    print(f"Ontology complete: {len(ontology):,} unique states")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    return ontology


def build_epistemology(
    paths: AtlasPaths, ontology: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    """
    Build epistemology as [N,256] indices using vectorized column-wise construction.

    Returns the epistemology array for use in phenomenology building.
    """
    print("Building epistemology...")
    n = int(ontology.size)

    from numpy.lib.format import open_memmap

    epi = open_memmap(
        str(paths.epistemology), mode="w+", dtype=np.uint32, shape=(n, 256)
    )

    assert np.all(ontology[:-1] <= ontology[1:]), "Ontology must be sorted"

    a = ((ontology >> 12) & LAYER_MASK_12).astype(np.uint32)
    b = (ontology & LAYER_MASK_12).astype(np.uint32)

    for byte in range(256):
        mask24 = int(XFORM_MASK_BY_BYTE[byte])
        m = (mask24 >> 12) & 0xFFF

        new_a = (b ^ 0xFFF).astype(np.uint32)
        new_b = ((a ^ m) ^ 0xFFF).astype(np.uint32)
        new_state = ((new_a << 12) | new_b).astype(np.uint32)

        idx = np.searchsorted(ontology, new_state).astype(np.int64)
        assert np.all(
            ontology[idx] == new_state
        ), f"Byte {byte}: closure violation detected"

        epi[:, byte] = idx.astype(np.uint32)

        if (byte + 1) % 64 == 0:
            epi.flush()
            print(
                f"  Processed {byte + 1:,} / 256 bytes"
                f" ({100 * (byte + 1) / 256:.1f}%)"
            )

    epi.flush()

    file_size = paths.epistemology.stat().st_size
    print(f"Epistemology complete: [{n:,}, 256] lookup table")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    return epi


def _intron_feature_matrix(intron_by_byte: NDArray[np.uint8]) -> NDArray[np.float32]:
    """
    Build [256, 10] feature matrix from intron-stage decomposition.

    This captures the byte's constitutional structure BEFORE expansion
    entangles it into the 12-bit mask geometry.

    Columns:
      0-5:  6-bit micro-reference as ±1 (dynamic payload, Appendix G FG/BG)
      6-7:  2-bit family index as ±1 (boundary anchors, Appendix G L0/LI)
      8:    cross-parity: (family XOR micro_ref low 2 bits) & 1, as ±1
      9:    total intron weight, normalized to [-1, +1] range

    The micro-reference (columns 0-5) carries the ONA/BU dynamic content.
    The family index (columns 6-7) carries the CS/UNA boundary identity.
    The cross-parity (column 8) captures their interaction.
    """
    F = np.zeros((256, 10), dtype=np.float32)
    for byte_val in range(256):
        intron = int(intron_by_byte[byte_val])
        micro = intron & 0x3F
        family = (intron >> 6) & 0x3

        # 6-bit micro-reference bits as ±1
        for i in range(6):
            F[byte_val, i] = 1.0 if ((micro >> i) & 1) else -1.0

        # 2-bit family index as ±1
        F[byte_val, 6] = 1.0 if (family & 1) else -1.0
        F[byte_val, 7] = 1.0 if ((family >> 1) & 1) else -1.0

        # Cross-parity: interaction between family and micro-reference
        F[byte_val, 8] = float((family ^ (micro & 0x3)) & 1) * 2.0 - 1.0

        # Total intron weight normalized to [-1, +1]
        w = bin(intron).count("1")
        F[byte_val, 9] = (w / 4.0) - 1.0

    return F


def _byte_feature_vector(
    byte: int, K: int, mask12_by_byte: NDArray[np.uint16]
) -> NDArray[np.float32]:
    """Compute deterministic byte feature vector for given K from mask_12 space."""
    m12 = int(mask12_by_byte[byte])

    bits12 = np.empty(12, dtype=np.float32)
    for i in range(12):
        bits12[i] = 1.0 if ((m12 >> i) & 1) else -1.0

    if K == 12:
        return bits12

    if K == 1:
        return np.array([np.mean(bits12)], dtype=np.float32)

    if K == 2:
        return np.array(
            [np.mean(bits12[0:6]), np.mean(bits12[6:12])], dtype=np.float32
        )

    if K == 3:
        row_groups = ((0, 1, 6, 7), (2, 3, 8, 9), (4, 5, 10, 11))
        out = np.zeros(3, dtype=np.float32)
        for r, idxs in enumerate(row_groups):
            out[r] = sum(bits12[i] for i in idxs) / 4.0
        return out

    if K == 4:
        return np.array(
            [
                np.mean(bits12[0:6:2]),
                np.mean(bits12[1:6:2]),
                np.mean(bits12[6:12:2]),
                np.mean(bits12[7:12:2]),
            ],
            dtype=np.float32,
        )

    if K == 6:
        frame_row_groups = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
        out = np.zeros(6, dtype=np.float32)
        for k, idxs in enumerate(frame_row_groups):
            out[k] = sum(bits12[i] for i in idxs) / 2.0
        return out

    if K == 8:
        frame_row_groups = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
        out = np.zeros(8, dtype=np.float32)
        for k, idxs in enumerate(frame_row_groups):
            out[k] = sum(bits12[i] for i in idxs) / 2.0
        out[6] = float(popcount(m12 & 0x03F) & 1) * 2.0 - 1.0
        out[7] = float(popcount((m12 >> 6) & 0x03F) & 1) * 2.0 - 1.0
        return out

    if K == 16:
        out = np.zeros(16, dtype=np.float32)
        out[:12] = bits12
        out[12] = float(popcount(m12) & 1) * 2.0 - 1.0
        out[13] = float(popcount(m12 & 0x03F) & 1) * 2.0 - 1.0
        out[14] = float(popcount((m12 >> 6) & 0x03F) & 1) * 2.0 - 1.0
        out[15] = float(vertex_charge_from_mask(m12) & 1) * 2.0 - 1.0
        return out

    raise ValueError(f"Unsupported K={K}")


def build_phenomenology(
    paths: AtlasPaths,
    ontology: NDArray[np.uint32],
    epistemology: NDArray[np.uint32],
) -> None:
    """
    Build the spectral atlas: phenomenology.npz

    Phenomenology is a structured archive containing six categories:

    1. KERNEL CONSTANTS
       Macro archetype values (ARCHETYPE_STATE24, A12, B12), GENE_MIC_S,
       transformation masks, dual code vectors.

    2. GENE_MIC INTRON-STAGE PRIORS
       The byte's constitutional structure BEFORE expansion into mask_12:
       intron decomposition, CGM stage parities, structured feature matrix.
       These preserve the byte boundary formalism that mask expansion consumes.

    3. PER-STATE OBSERVABLES
       state_horizon, state_vertex — properties of ontology states.

    4. SPECTRAL PHASE CUBE
       phase[N, 256] — cycle position of each state under each byte permutation.

    5. BACKWARD-PASS OBSERVABLES
       next_horizon, next_vertex, next_phase — pre-computed lookahead arrays.

    6. PER-BYTE MASK-STAGE HELPERS
       mask12_by_byte, byte_weight, byte_charge, features_K* —
       derived from the 12-bit projected masks (post-expansion).

    Categories 2 and 6 both describe bytes but from different stages:
    - Category 2: byte BEFORE projection (intron space, CGM priors preserved)
    - Category 6: byte AFTER projection (mask space, geometry-compatible)
    """
    print("Building phenomenology (spectral atlas)...")
    n = len(ontology)

    # =================================================================
    # Category 2: GENE_Mic intron-stage priors
    # =================================================================
    print("  Computing GENE_Mic intron-stage priors...")

    # Primary decomposition (Appendix G formalism)
    intron_by_byte = np.array(
        [(b ^ GENE_MIC_S) for b in range(256)], dtype=np.uint8
    )
    micro_ref_by_byte = (intron_by_byte & 0x3F).astype(np.uint8)
    family_by_byte = ((intron_by_byte >> 6) & 0x3).astype(np.uint8)

    # CGM stage parities (Appendix G palindromic structure)
    # bit 0,7 = L0 (CS anchor); bit 1,6 = LI (UNA chirality)
    # bit 2,5 = FG (ONA dynamics); bit 3,4 = BG (BU balance)
    intron_16 = intron_by_byte.astype(np.uint16)
    L0_parity = (((intron_16) & 1) ^ ((intron_16 >> 7) & 1)).astype(np.uint8)
    LI_parity = (((intron_16 >> 1) & 1) ^ ((intron_16 >> 6) & 1)).astype(
        np.uint8
    )
    FG_parity = (((intron_16 >> 2) & 1) ^ ((intron_16 >> 5) & 1)).astype(
        np.uint8
    )
    BG_parity = (((intron_16 >> 3) & 1) ^ ((intron_16 >> 4) & 1)).astype(
        np.uint8
    )

    # Structured intron feature matrix [256, 10]
    intron_features = _intron_feature_matrix(intron_by_byte)

    # Verification: 64 unique micro-references, 4 families, 256 unique introns
    assert len(np.unique(micro_ref_by_byte)) == 64, (
        "Expected 64 unique micro-references"
    )
    assert set(family_by_byte.tolist()) == {0, 1, 2, 3}, (
        "Expected 4 families {0,1,2,3}"
    )
    assert len(np.unique(intron_by_byte)) == 256, "Expected 256 unique introns"

    print(
        f"    Intron decomposition: 256 introns = "
        f"64 micro-refs × 4 families"
    )
    print(
        f"    CGM stage parities: L0, LI, FG, BG "
        f"(Appendix G palindromic structure)"
    )
    print(f"    Intron feature matrix: {intron_features.shape}")

    # =================================================================
    # Category 6: Per-byte mask-stage helpers
    # =================================================================

    # Build mask_to_byte lookup
    mask_to_byte: dict[int, int] = {}
    for b in range(256):
        m12 = mask12_for_byte(b)
        mask_to_byte[m12] = b

    # Build per-byte helpers from mask_12 space
    mask12_by_byte = np.array(
        [mask12_for_byte(b) for b in range(256)], dtype=np.uint16
    )
    byte_weight = np.array(
        [popcount(mask12_by_byte[b]) for b in range(256)], dtype=np.uint8
    )
    byte_charge = np.array(
        [vertex_charge_from_mask(mask12_by_byte[b]) for b in range(256)],
        dtype=np.uint8,
    )

    # =================================================================
    # Category 3: Per-state observables
    # =================================================================
    print("  Computing per-state observables...")
    state_horizon = np.empty(n, dtype=np.uint8)
    state_vertex = np.empty(n, dtype=np.uint8)

    for i, s24 in enumerate(ontology):
        a12 = (int(s24) >> 12) & 0xFFF
        mask = a12 ^ ARCHETYPE_A12
        try:
            h = mask_to_byte[mask]
        except KeyError:
            raise RuntimeError(
                f"Mask {mask:#x} not in mask_to_byte; atlas inconsistent"
            )
        state_horizon[i] = h
        state_vertex[i] = vertex_charge_from_mask(mask)

    # =================================================================
    # Category 4: Spectral phase cube
    # =================================================================
    print("  Computing spectral phase cube...")
    phase = np.zeros((n, 256), dtype=np.uint8)

    for b in range(256):
        perm = epistemology[:, b]
        seen = np.zeros(n, dtype=np.uint8)

        for start in range(n):
            if seen[start]:
                continue

            cycle: list[int] = []
            x = int(start)
            while not seen[x]:
                seen[x] = 1
                cycle.append(x)
                x = int(perm[x])

            anchor = min(cycle)
            k = cycle.index(anchor)
            cycle = cycle[k:] + cycle[:k]

            cycle_len = len(cycle)

            if b != GENE_MIC_S:
                if cycle_len != 4:
                    raise RuntimeError(
                        f"Byte {b} produced cycle length {cycle_len},"
                        f" expected 4"
                    )
            else:
                if cycle_len not in (1, 2):
                    raise RuntimeError(
                        f"Reference byte {b} produced cycle length"
                        f" {cycle_len}, expected 1 or 2"
                    )

            for p, s in enumerate(cycle):
                phase[s, b] = p

        if (b + 1) % 64 == 0:
            print(f"    Phase computed for {b + 1:,} / 256 bytes")

        max_phase = int(phase[:, b].max())
        if b != GENE_MIC_S:
            if max_phase != 3:
                raise RuntimeError(
                    f"Phase max for byte {b} is {max_phase}, expected 3"
                )
        else:
            if max_phase > 1:
                raise RuntimeError(
                    f"Phase max for reference byte {b} is {max_phase},"
                    f" expected <= 1"
                )

    # =================================================================
    # Category 5: Backward-pass observables
    # =================================================================
    print("  Computing backward-pass observables...")
    next_horizon = state_horizon[epistemology]
    next_vertex = state_vertex[epistemology]

    next_phase = np.empty((n, 256), dtype=np.uint8)
    for b in range(256):
        next_phase[:, b] = phase[epistemology[:, b], b]

    # =================================================================
    # Category 6 continued: Byte feature matrices from mask_12 space
    # =================================================================
    print("  Computing byte feature matrices (mask-stage)...")
    K_VALUES = (1, 2, 3, 4, 6, 8, 12, 16)
    feature_matrices: dict[str, NDArray[np.float32]] = {}

    for K in K_VALUES:
        F = np.zeros((256, K), dtype=np.float32)
        for b in range(256):
            F[b, :] = _byte_feature_vector(b, K, mask12_by_byte)
        feature_matrices[f"features_K{K}"] = F

    # =================================================================
    # Save all categories to phenomenology.npz
    # =================================================================
    print("  Saving phenomenology.npz...")
    np.savez(
        paths.phenomenology,
        # === Category 1: Kernel constants ===
        atlas_version=ATLAS_VERSION,
        archetype_state24=np.uint32(ARCHETYPE_STATE24),
        archetype_a12=np.uint16(ARCHETYPE_A12),
        archetype_b12=np.uint16(ARCHETYPE_B12),
        gene_mic_s=np.uint8(GENE_MIC_S),
        xform_mask_by_byte=XFORM_MASK_BY_BYTE,
        c_perp_12=np.array(C_PERP_12, dtype=np.uint16),
        q0=np.uint16(Q0),
        q1=np.uint16(Q1),
        # === Category 2: GENE_Mic intron-stage priors ===
        intron_by_byte=intron_by_byte,
        micro_ref_by_byte=micro_ref_by_byte,
        family_by_byte=family_by_byte,
        L0_parity=L0_parity,
        LI_parity=LI_parity,
        FG_parity=FG_parity,
        BG_parity=BG_parity,
        intron_features=intron_features,
        # === Category 3: Per-state observables ===
        state_horizon=state_horizon,
        state_vertex=state_vertex,
        # === Category 4: Spectral phase cube ===
        phase=phase,
        # === Category 5: Backward-pass observables ===
        next_horizon=next_horizon,
        next_vertex=next_vertex,
        next_phase=next_phase,
        # === Category 6: Per-byte mask-stage helpers ===
        mask12_by_byte=mask12_by_byte,
        byte_weight=byte_weight,
        byte_charge=byte_charge,
        **feature_matrices,  # pyright: ignore[reportArgumentType]
    )

    file_size = paths.phenomenology.stat().st_size
    print("Phenomenology complete: spectral atlas")
    print(
        f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)"
    )
    print(f"  Atlas version: {ATLAS_VERSION}")
    print("  Categories:")
    print("    1. Kernel constants")
    print("    2. GENE_Mic intron-stage priors (6-bit micro-ref + 2-bit family"
          " + CGM parities)")
    print("    3. Per-state observables (horizon, vertex)")
    print("    4. Spectral phase cube")
    print("    5. Backward-pass observables (next_horizon/vertex/phase)")
    print("    6. Per-byte mask-stage helpers (mask12, weight, charge,"
          " features_K*)")


def build_all(base_dir: Path) -> None:
    paths = AtlasPaths(base=base_dir)
    print("Atlas Builder v2.2")
    print("==================")
    t0 = time.time()

    ontology = build_ontology(paths)
    print("")

    epistemology = build_epistemology(paths, ontology)
    print("")

    build_phenomenology(paths, ontology, epistemology)
    print("")

    elapsed = time.time() - t0

    total_size = (
        paths.ontology.stat().st_size
        + paths.epistemology.stat().st_size
        + paths.phenomenology.stat().st_size
    )

    print("Summary")
    print("=======")
    print(f"Total build time: {elapsed:.2f} seconds")
    print(
        f"Total atlas size: {total_size:,} bytes"
        f" ({total_size / 1024 / 1024:.2f} MB)"
    )
    print(f"Output directory: {paths.base}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build atlas: ontology.npy, epistemology.npy,"
        " phenomenology.npz"
    )
    parser.add_argument("--out", type=Path, default=Path("data/atlas"))
    args = parser.parse_args()
    build_all(args.out)


if __name__ == "__main__":
    main()