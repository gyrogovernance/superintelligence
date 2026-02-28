#!/usr/bin/env python3
"""
src/tools/validate_composite.py

Tape Recorder for the unified 4-Layer Stack.

Parses a corpus through L1/L2/L3/L4 as a SINGLE system.
Records: composite_state → {observed bytes}
Measures: how layers COLLABORATE to constrain byte selection.

Key metric: "Given my position in ALL layers simultaneously,
how many bytes has the corpus produced from this exact coordinate?"

This is the branching factor of the geometric tube that language
carves through the 4-layer manifold.

Usage:
    python -m src.tools.validate_composite --corpus path/to/corpus.txt
    python src/tools/validate_composite.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.router.constants import GENE_MIC_S, mask12_for_byte, popcount
from src.tools.layers import L4State, create_default_four_layers


# =====================================================================
# Corpus loading
# =====================================================================


def load_corpus_bytes(path: Path, max_bytes: int) -> bytes:
    """Load bytes from file. JSONL: concatenate text from known fields."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() == ".jsonl":
        lines = data.decode("utf-8", errors="ignore").strip().split("\n")
        parts: List[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                for key in ("question", "answer", "context", "text"):
                    if key in rec and rec[key]:
                        parts.append(str(rec[key]))
            except json.JSONDecodeError:
                parts.append(line)
        text = " ".join(parts)
        data = text.encode("utf-8")
    return data[:max_bytes]


# =====================================================================
# L4 coarse banding
# =====================================================================


def l4_band(l4: L4State) -> int:
    """Bin L4 closure defect into 4 bands. 0=near closure, 3=far."""
    w = popcount(l4.O & 0xFFF) + popcount(l4.E & 0xFFF)
    return min(w // 6, 3)


# =====================================================================
# Tape recording structure
# =====================================================================


class TapeRecording:
    """Records composite_state → {observed bytes} with visit counts."""

    def __init__(self):
        self.byte_sets: Dict[Any, set] = defaultdict(set)
        self.visit_counts: Dict[Any, int] = defaultdict(int)

    def record(self, key: Any, byte_val: int) -> None:
        self.byte_sets[key].add(byte_val)
        self.visit_counts[key] += 1

    @property
    def unique_states(self) -> int:
        return len(self.byte_sets)

    def branching_factors(self) -> np.ndarray:
        """BF per visit (each visit gets the BF of its state)."""
        bfs: List[int] = []
        for key, byte_set in self.byte_sets.items():
            bf = len(byte_set)
            visits = self.visit_counts[key]
            bfs.extend([bf] * visits)
        return np.array(bfs, dtype=np.int32)

    def singleton_visits(self) -> int:
        """Visits where BF=1 (byte uniquely determined by state)."""
        return sum(v for k, v in self.visit_counts.items() if len(self.byte_sets[k]) == 1)


# =====================================================================
# Granularity definitions
# =====================================================================

GRANULARITIES = [
    "L1",
    "L2",
    "L3",
    "L1+L2",
    "L1+L3",
    "L2+L3",
    "L1+L2+L3",
    "L1+L2+L3+L4b",
]


def make_key(gran: str, l1: int, l2: int, l3: int, l4b: int) -> Any:
    if gran == "L1":
        return l1
    if gran == "L2":
        return l2
    if gran == "L3":
        return l3
    if gran == "L1+L2":
        return (l1, l2)
    if gran == "L1+L3":
        return (l1, l3)
    if gran == "L2+L3":
        return (l2, l3)
    if gran == "L1+L2+L3":
        return (l1, l2, l3)
    if gran == "L1+L2+L3+L4b":
        return (l1, l2, l3, l4b)
    raise ValueError(f"Unknown granularity: {gran}")


# =====================================================================
# Tape recorder: single unified walk
# =====================================================================


def run_tape_recorder(
    corpus_path: Path,
    layers,
    max_bytes: int,
) -> Dict[str, TapeRecording]:
    """
    Walk corpus through unified 4-layer stack.
    At each step, the composite state (L1, L2, L3, L4) is ONE coordinate.
    Record which byte was observed from that coordinate.
    """
    data = load_corpus_bytes(corpus_path, max_bytes)
    n = len(data)

    recordings = {g: TapeRecording() for g in GRANULARITIES}
    regs = layers.regs
    regs.reset()

    for pos in range(n):
        if pos % 50000 == 0 and pos > 0:
            print(f"  Recording: {pos:,}/{n:,}...", end="\r")

        byte_val = data[pos]

        # Read composite coordinate BEFORE this byte
        l1 = regs.l1_state8
        l2 = regs.l2_state16
        l3 = regs.l3_state24
        l4b = l4_band(regs.l4)

        # Record observation at every granularity
        for gran in GRANULARITIES:
            key = make_key(gran, l1, l2, l3, l4b)
            recordings[gran].record(key, byte_val)

        # Step the UNIFIED stack (all layers move together)
        layers.ingest_byte(byte_val)

    print(f"  Recording: {n:,}/{n:,} done.")
    return recordings


# =====================================================================
# Byte formalism analysis
# =====================================================================


def byte_formalism_stats(corpus_path: Path, max_bytes: int) -> Dict[str, Any]:
    """Analyze byte-level structure: families, micro-refs, mask coverage."""
    data = load_corpus_bytes(corpus_path, max_bytes)
    n = len(data)

    family_counts = [0, 0, 0, 0]
    micro_ref_seen: set = set()
    mask12_seen: set = set()
    intron_seen: set = set()

    for byte_val in data:
        intron = (byte_val ^ GENE_MIC_S) & 0xFF
        family = (intron >> 6) & 0x3
        micro_ref = intron & 0x3F
        m12 = mask12_for_byte(byte_val) & 0xFFF

        family_counts[family] += 1
        micro_ref_seen.add(micro_ref)
        mask12_seen.add(m12)
        intron_seen.add(intron)

    return {
        "family_counts": family_counts,
        "family_pcts": [100.0 * c / n for c in family_counts],
        "micro_refs_used": len(micro_ref_seen),
        "mask12_used": len(mask12_seen),
        "introns_used": len(intron_seen),
        "total_bytes": n,
    }


# =====================================================================
# Report
# =====================================================================


def print_report(
    recordings: Dict[str, TapeRecording],
    corpus_path: Path,
    n_bytes: int,
    bf_stats: Dict[str, Any],
):
    print()
    print("=" * 80)
    print("  COLLABORATIVE CONSTRAINT ANALYSIS — Tape Recorder")
    print("=" * 80)
    print(f"  Corpus:  {corpus_path}")
    print(f"  Bytes:   {n_bytes:,}")
    print()

    # --- Main table ---
    header = (
        f"  {'Granularity':<20}"
        f"{'Unique':>9}"
        f"{'Mean BF':>9}"
        f"{'Med BF':>8}"
        f"{'BF=1':>7}"
        f"{'BF≤3':>7}"
        f"{'BF≤5':>7}"
        f"{'BF≤10':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for gran in GRANULARITIES:
        rec = recordings[gran]
        bfs = rec.branching_factors()
        if len(bfs) == 0:
            continue

        unique = rec.unique_states
        mean_bf = float(bfs.mean())
        med_bf = float(np.median(bfs))
        bf1 = 100.0 * np.sum(bfs == 1) / len(bfs)
        bf3 = 100.0 * np.sum(bfs <= 3) / len(bfs)
        bf5 = 100.0 * np.sum(bfs <= 5) / len(bfs)
        bf10 = 100.0 * np.sum(bfs <= 10) / len(bfs)

        print(
            f"  {gran:<20}"
            f"{unique:>9,}"
            f"{mean_bf:>9.2f}"
            f"{med_bf:>8.1f}"
            f"{bf1:>6.1f}%"
            f"{bf3:>6.1f}%"
            f"{bf5:>6.1f}%"
            f"{bf10:>6.1f}%"
        )

    # --- Layer correlation ---
    print()
    print("  LAYER CORRELATION")
    print("  " + "-" * 50)

    l1u = recordings["L1"].unique_states
    l2u = recordings["L2"].unique_states
    l3u = recordings["L3"].unique_states
    l1l2u = recordings["L1+L2"].unique_states
    l1l3u = recordings["L1+L3"].unique_states
    l2l3u = recordings["L2+L3"].unique_states
    allu = recordings["L1+L2+L3"].unique_states
    all4u = recordings["L1+L2+L3+L4b"].unique_states

    print(f"  L1 unique:            {l1u:>9,}")
    print(f"  L2 unique:            {l2u:>9,}")
    print(f"  L3 unique:            {l3u:>9,}")
    print(f"  (L1,L2) pairs:        {l1l2u:>9,}")
    print(f"  (L1,L3) pairs:        {l1l3u:>9,}")
    print(f"  (L2,L3) pairs:        {l2l3u:>9,}")
    print(f"  (L1,L2,L3) triples:   {allu:>9,}")
    print(f"  (L1,L2,L3,L4b) quads: {all4u:>9,}")

    if l2l3u == l2u and l2u == l3u:
        print("  → L2 and L3 are PERFECTLY COUPLED")
    elif l2l3u > max(l2u, l3u):
        extra = l2l3u - max(l2u, l3u)
        print(f"  → L2+L3 adds {extra:,} extra distinctions")

    if allu > l2l3u:
        extra = allu - l2l3u
        print(f"  → Adding L1 contributes {extra:,} extra distinctions")

    if all4u > allu:
        extra = all4u - allu
        print(f"  → Adding L4 band contributes {extra:,} extra distinctions")

    # --- BF distribution for most constrained ---
    best_gran = GRANULARITIES[-1]
    bfs = recordings[best_gran].branching_factors()
    print()
    print(f"  BF DISTRIBUTION — {best_gran}")
    print("  " + "-" * 50)
    for threshold in [1, 2, 3, 5, 10, 20, 50, 100]:
        count = int(np.sum(bfs <= threshold))
        pct = 100.0 * count / len(bfs)
        bar = "█" * int(pct / 2)
        print(f"    BF≤{threshold:>3}: {count:>7,} ({pct:>5.1f}%) {bar}")
    print(f"    Max BF:  {int(bfs.max())}")
    print(f"    Min BF:  {int(bfs.min())}")

    # --- Byte formalism ---
    print()
    print("  BYTE FORMALISM")
    print("  " + "-" * 50)
    for i in range(4):
        print(
            f"    Family {i}: {bf_stats['family_counts'][i]:>7,}"
            f" ({bf_stats['family_pcts'][i]:>5.1f}%)"
        )
    print(f"    Micro-refs used:  {bf_stats['micro_refs_used']}/64")
    print(f"    Mask12 used:      {bf_stats['mask12_used']}/256")
    print(f"    Introns used:     {bf_stats['introns_used']}/256")

    # --- Summary ---
    l3_bf = float(recordings["L3"].branching_factors().mean())
    all_bf = float(recordings["L1+L2+L3"].branching_factors().mean())
    all4_bf = float(recordings["L1+L2+L3+L4b"].branching_factors().mean())

    print()
    print("  COLLABORATION SUMMARY")
    print("  " + "-" * 50)
    print(f"    L3 alone:        mean BF = {l3_bf:.2f}")
    print(f"    L1+L2+L3:        mean BF = {all_bf:.2f}")
    print(f"    L1+L2+L3+L4b:    mean BF = {all4_bf:.2f}")

    if l3_bf > 0:
        reduction = 100.0 * (1.0 - all4_bf / l3_bf)
        print(f"    → Full collaboration reduces BF by {reduction:.1f}%")


# =====================================================================
# Entry point
# =====================================================================


def main():
    root = Path(_root)
    parser = argparse.ArgumentParser(
        description="Tape recorder: measure collaborative 4-layer constraints"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "docs" / "notes" / "cgm_dataset_main.jsonl",
        help="Path to corpus (text or .jsonl)",
    )
    parser.add_argument(
        "--l3-path",
        type=Path,
        default=root / "data" / "layers" / "l3_packed_u24.bin",
        help="Path to L3 FSM table",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=200_000,
        help="Maximum bytes to analyze",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    print("Initializing 4-layer stack...")
    layers = create_default_four_layers(
        l3_path=args.l3_path,
        build_l3_if_missing=False,
    )

    print(f"Corpus: {args.corpus}")
    recordings = run_tape_recorder(args.corpus, layers, args.max_bytes)

    n = sum(recordings["L1"].visit_counts.values())
    bf_stats = byte_formalism_stats(args.corpus, args.max_bytes)

    print_report(recordings, args.corpus, n, bf_stats)


if __name__ == "__main__":
    main()