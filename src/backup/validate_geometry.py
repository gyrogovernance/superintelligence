#!/usr/bin/env python3
"""
src/tools/validate_geometry.py

Reports statistics from the 4-layer stack (L1/L2/L3/L4) when reading a corpus.
Uses only layers + first-principles constants; no atlas/ontology/epistemology.

Usage:
    python src/tools/validate_geometry.py
    python -m src.tools.validate_geometry --corpus path/to/corpus.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on path for direct execution
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

import numpy as np

from src.tools.layers import create_default_four_layers
from src.router.constants import (
    GENE_MIC_S,
    mask12_for_byte,
    popcount,
)


def get_family(byte: int) -> int:
    """Extract 2-bit family index from byte (intron bits 6,7)."""
    intron = (byte ^ GENE_MIC_S) & 0xFF
    return (intron >> 6) & 0x3


def load_corpus_bytes(path: Path, max_bytes: int) -> bytes:
    """Load bytes from file. JSONL: concatenate text from question, answer, context."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() == ".jsonl":
        lines = data.decode("utf-8", errors="ignore").strip().split("\n")
        parts = []
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                for key in ("question", "answer", "context"):
                    if key in rec and rec[key]:
                        parts.append(str(rec[key]))
            except json.JSONDecodeError:
                parts.append(line)
        text = " ".join(parts)
        data = text.encode("utf-8")
    return data[:max_bytes]


class UTF8Validator:
    """L1 constraint: Valid UTF-8 byte sequences only."""

    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0

    def is_valid(self, byte: int) -> bool:
        b = byte & 0xFF
        if self.state == 0:
            if b < 0x80:
                return True
            elif 0xC0 <= b <= 0xDF:
                self.state = 1
                return True
            elif 0xE0 <= b <= 0xEF:
                self.state = 2
                return True
            elif 0xF0 <= b <= 0xF7:
                self.state = 3
                return True
            return False
        elif self.state == 1:
            if 0x80 <= b <= 0xBF:
                self.state = 0
                return True
            return False
        elif self.state == 2:
            if 0x80 <= b <= 0xBF:
                self.state = 1
                return True
            return False
        elif self.state == 3:
            if 0x80 <= b <= 0xBF:
                self.state = 2
                return True
            return False
        return False

    def copy(self) -> "UTF8Validator":
        v = UTF8Validator()
        v.state = self.state
        return v


@dataclass
class CorpusStats:
    """Aggregated statistics from 4-layer corpus walk."""

    total_bytes: int = 0
    l1_bf_mean: float = 0.0
    l1_bf_median: float = 0.0
    l1_gold_in_valid_pct: float = 0.0
    l1_state_counts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    l2_states_unique: int = 0
    l3_states_unique: int = 0
    l4_closure_events: int = 0
    l4_closure_positions: List[int] = field(default_factory=list)
    l4_defect_samples: List[float] = field(default_factory=list)
    mask12_unique: int = 0
    family_counts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    intron_histogram: List[int] = field(default_factory=lambda: [0] * 256)
    byte_histogram: List[int] = field(default_factory=lambda: [0] * 256)


def analyze_corpus(
    corpus_path: Path,
    layers,
    max_bytes: int = 200_000,
    sample_defect_every: int = 500,
) -> CorpusStats:
    """
    Walk corpus byte-by-byte through the 4 layers.
    Collect: L1 branching factors, L2/L3 state diversity, L4 closure events,
    family distribution, intron histogram.
    """
    data = load_corpus_bytes(corpus_path, max_bytes)
    n = len(data)

    stats = CorpusStats(total_bytes=n)
    utf8_val = UTF8Validator()
    regs = layers.regs
    regs.reset()

    l2_seen: Set[int] = set()
    l3_seen: Set[int] = set()
    mask12_seen: Set[int] = set()
    l1_bfs: List[int] = []
    gold_valid_count = 0

    for pos, gold_byte in enumerate(data):
        if pos % 50000 == 0 and pos > 0:
            print(f"  Position {pos}/{n}...", end="\r")

        # L1 branching factor: how many UTF-8-valid next bytes
        valid_count = 0
        for b in range(256):
            test_utf8 = utf8_val.copy()
            if test_utf8.is_valid(b):
                valid_count += 1
                if b == gold_byte:
                    gold_valid_count += 1
        l1_bfs.append(valid_count)

        # Record layer states
        l2_seen.add(regs.l2_state16)
        l3_seen.add(regs.l3_state24)

        # L1 state (0=start, 1-3=expecting continuation)
        stats.l1_state_counts[min(utf8_val.state, 3)] += 1

        # L4 closure (identity cycle: O=0, E=0, parity=0)
        if (regs.l4.O == 0) and (regs.l4.E == 0) and (regs.l4.parity == 0):
            stats.l4_closure_events += 1
            stats.l4_closure_positions.append(pos)

        mask12_seen.add(mask12_for_byte(gold_byte) & 0xFFF)

        if pos % sample_defect_every == 0:
            defect = (popcount(regs.l4.O & 0xFFF) + popcount(regs.l4.E & 0xFFF)) / 24.0
            stats.l4_defect_samples.append(defect)

        # Byte formalism
        stats.byte_histogram[gold_byte] += 1
        intron_val = (gold_byte ^ GENE_MIC_S) & 0xFF
        stats.intron_histogram[intron_val] += 1
        stats.family_counts[get_family(gold_byte)] += 1

        # Step layers with gold byte
        utf8_val.is_valid(gold_byte)
        layers.ingest_byte(gold_byte)

    stats.l2_states_unique = len(l2_seen)
    stats.l3_states_unique = len(l3_seen)
    stats.mask12_unique = len(mask12_seen)
    stats.l1_bf_mean = float(np.mean(l1_bfs)) if l1_bfs else 0.0
    stats.l1_bf_median = float(np.median(l1_bfs)) if l1_bfs else 0.0
    stats.l1_gold_in_valid_pct = 100.0 * gold_valid_count / n if n else 0.0

    return stats


def print_report(stats: CorpusStats, corpus_path: Path):
    """Print formatted report of 4-layer corpus statistics."""
    print()
    print("-" * 50)
    print("4-LAYER CORPUS STATISTICS")
    print("-" * 50)
    print(f"Corpus: {corpus_path}")
    print(f"Bytes analyzed: {stats.total_bytes:,}")

    print("\n--- L1 (UTF-8) ---")
    print(f"L1 branching factor (valid next bytes): mean={stats.l1_bf_mean:.1f}, "
          f"median={stats.l1_bf_median:.1f}")
    print(f"Gold byte in L1-valid set: {stats.l1_gold_in_valid_pct:.1f}%")
    total_l1 = sum(stats.l1_state_counts)
    if total_l1:
        for i, c in enumerate(stats.l1_state_counts):
            lbl = "start" if i == 0 else f"cont{i}"
            print(f"L1 state {lbl}: {100.0*c/total_l1:.1f}%")

    print("\n--- L2 (Chirality) ---")
    pct = 100.0 * stats.l2_states_unique / 65536
    print(f"Unique L2 states: {stats.l2_states_unique:,} / 65536 ({pct:.1f}%)")

    print("\n--- L3 (24-bit GENE_Mac) ---")
    pct = 100.0 * stats.l3_states_unique / 16777216
    print(f"Unique L3 states: {stats.l3_states_unique:,} / 16777216 ({pct:.2f}%)")

    print("\n--- L4 (Closure) ---")
    print(f"Identity-cycle events (O=0,E=0,parity=0): {stats.l4_closure_events}")
    if stats.l4_closure_positions:
        pos_str = ",".join(str(p) for p in stats.l4_closure_positions[:8])
        if len(stats.l4_closure_positions) > 8:
            pos_str += ",..."
        print(f"Closure positions: {pos_str}")
    if stats.l4_defect_samples:
        arr = np.array(stats.l4_defect_samples)
        print(f"L4 defect (sampled): mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")
        hist, _ = np.histogram(arr, bins=[0, 0.25, 0.5, 0.75, 1.0])
        print(f"L4 defect distribution: [0-0.25)={hist[0]}, [0.25-0.5)={hist[1]}, "
              f"[0.5-0.75)={hist[2]}, [0.75-1]={hist[3]}")

    print("\n--- Mask12 (12-bit mask space) ---")
    print(f"Unique mask12 values: {stats.mask12_unique} / 4096")

    print("\n--- Byte Formalism (intron = byte XOR 0xAA) ---")
    total_fam = sum(stats.family_counts)
    for i, c in enumerate(stats.family_counts):
        pct = 100.0 * c / total_fam if total_fam else 0
        print(f"Family {i}: {c:,} ({pct:.1f}%)")
    non_zero = sum(1 for h in stats.intron_histogram if h > 0)
    print(f"Intron values used: {non_zero}/256")
    top_b = sorted(range(256), key=lambda i: stats.byte_histogram[i], reverse=True)[:5]
    top_i = sorted(range(256), key=lambda i: stats.intron_histogram[i], reverse=True)[:5]
    print("Top 5 bytes (raw):   " + " ".join(f"0x{b:02x}({stats.byte_histogram[b]})" for b in top_b))
    print("Top 5 introns:       " + " ".join(f"0x{i:02x}({stats.intron_histogram[i]})" for i in top_i))


def main():
    root = Path(_root)
    parser = argparse.ArgumentParser(
        description="Report 4-layer statistics when reading a corpus"
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
    parser.add_argument(
        "--sample-defect-every",
        type=int,
        default=500,
        help="Sample L4 defect every N bytes",
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

    print(f"Analyzing corpus: {args.corpus}")
    stats = analyze_corpus(
        args.corpus,
        layers,
        max_bytes=args.max_bytes,
        sample_defect_every=args.sample_defect_every,
    )
    print_report(stats, args.corpus)


if __name__ == "__main__":
    main()
