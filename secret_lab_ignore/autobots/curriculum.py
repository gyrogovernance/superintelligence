"""
FSM curriculum generator.

Generates training trajectories by walking the L1/L2/L3 FSMs.
This is NOT text prediction training - it is structural learning.

IMPORTANT: This file imports from src.tools.layers (the 13GB L3 table).
           It runs ONLY during training data generation.
           It is NOT needed at inference.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from . import physics


def _ensure_layers() -> Any:
    """Lazy import to avoid loading L3 table at module level."""
    from src.tools.layers import (
        Layer1FSM,
        Layer2FSM,
        Layer3FSM,
        Layer4,
        create_default_four_layers,
    )
    return create_default_four_layers, Layer1FSM, Layer2FSM, Layer3FSM, Layer4


class FSMCurriculum:
    def __init__(self, l3_path: Path):
        create_default_four_layers, _, _, _, _ = _ensure_layers()
        self.four = create_default_four_layers(
            l3_path=Path(l3_path),
            build_l3_if_missing=False,
        )

    def generate_random_walks(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Random walks from random starting states."""
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            seq = []
            for _ in range(seq_len):
                b = rng.randint(0, 255)
                seq.append(b)
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_family_balanced(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Walks that balance all 4 families equally."""
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            seq = []
            for step in range(seq_len):
                fam = step % 4
                micro = rng.randint(0, 63)
                intron = (fam << 6) | micro
                b = intron ^ physics.GENE_MIC_S
                seq.append(b & 0xFF)
                self.four.ingest_byte(seq[-1])
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_repeat_patterns(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Simple periodic patterns. Not language.
        Purpose: force temporal predictability so attention + L4 can learn.
        Periods {1,2,4} align with L1 memory and P7.
        """
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            period = rng.choice([1, 2, 4])
            pattern = [rng.randint(0, 255) for _ in range(period)]
            seq = [pattern[i % period] for i in range(seq_len)]
            for b in seq:
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_family_locked(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Lock to a single family (2-bit anchor). Micro varies.
        Purpose: family becomes strongly predictable; head must use family branch.
        """
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            fam = rng.randint(0, 3)
            seq = []
            for _ in range(seq_len):
                micro = rng.randint(0, 63)
                intr = (fam << 6) | micro
                b = (intr ^ physics.GENE_MIC_S) & 0xFF
                seq.append(b)
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_micro_locked(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Lock to a single micro (6-bit payload). Family varies.
        Purpose: micro becomes strongly predictable; head must use micro branch.
        """
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            micro = rng.randint(0, 63)
            seq = []
            for _ in range(seq_len):
                fam = rng.randint(0, 3)
                intr = (fam << 6) | micro
                b = (intr ^ physics.GENE_MIC_S) & 0xFF
                seq.append(b)
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_vertex_locked(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Lock to a single vertex charge class (K4 quotient).
        Purpose: TL stream + vertex signal must become predictive.
        """
        rng = random.Random(seed)
        vertex_bytes = {v: [] for v in range(4)}
        for b in range(256):
            m12 = physics.expand_intron_to_mask12(physics.intron(b))
            v = physics.vertex_charge(m12)
            vertex_bytes[v].append(b)

        out = []
        for _ in range(num_sequences):
            self.four.reset()
            vtx = rng.randint(0, 3)
            pool = vertex_bytes[vtx]
            seq = [rng.choice(pool) for _ in range(seq_len)]
            for b in seq:
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_separator_patterns(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Separator lemma exposure: alternating content bytes with 0xAA.
        Purpose: teach reference-byte routing behavior.
        """
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            seq = []
            for _ in range((seq_len + 1) // 2):
                content = rng.randint(0, 255)
                seq.extend([content, 0xAA])
            seq = seq[:seq_len]
            for b in seq:
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_horizon_walks(
        self,
        num_sequences: int,
        seq_len: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Directed transport w.r.t. horizon distance (canonical observable).
        Pure physics policy: first half minimize HD, second half maximize HD.
        Tie-break: smallest byte value. No hidden candidate RNG.
        """
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            state24 = physics.ARCHETYPE_STATE24
            seq = []
            for t in range(seq_len):
                toward_horizon = t < seq_len // 2
                best_b = 0
                best_hd = 13 if toward_horizon else -1
                for b in range(256):
                    ns = physics.step_state_l3_scalar(state24, b)
                    na = (ns >> 12) & 0xFFF
                    nb = ns & 0xFFF
                    hd = ((na ^ (nb ^ 0xFFF)) & 0xFFF).bit_count()
                    if toward_horizon:
                        if hd < best_hd:
                            best_hd = hd
                            best_b = b
                    else:
                        if hd > best_hd:
                            best_hd = hd
                            best_b = b
                seq.append(best_b)
                state24 = physics.step_state_l3_scalar(state24, best_b)
                self.four.ingest_byte(best_b)

            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_closure_walks(
        self,
        num_sequences: int,
        seq_len: int = 64,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Sustained alternation to expose P7-style structure: xyxyxyxy..."""
        rng = random.Random(seed)
        out = []
        for _ in range(num_sequences):
            self.four.reset()
            x = rng.randint(0, 255)
            y = rng.randint(0, 255)
            while y == x:
                y = rng.randint(0, 255)
            seq = [x if (i % 2 == 0) else y for i in range(seq_len)]
            for b in seq:
                self.four.ingest_byte(b)
            regs = self.four.regs
            out.append({
                "input_ids": seq,
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
            })
        return out

    def generate_p7_contrastive(
        self,
        num_samples: int,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Generate pairs for P7 (closure) learning:
        - Positive: xyxy (closes to identity, L4 resets)
        - Negative: xyxz (does not close, L4 drifts)
        """
        rng = random.Random(seed)
        out = []
        for _ in range(num_samples):
            x = rng.randint(0, 255)
            y = rng.randint(0, 255)
            z = rng.randint(0, 255)
            while z == y:
                z = rng.randint(0, 255)
            for seq, label in [([x, y, x, y], 1), ([x, y, x, z], 0)]:
                self.four.reset()
                for b in seq:
                    self.four.ingest_byte(b)
                regs = self.four.regs
                out.append({
                    "input_ids": seq,
                    "p7_closure": label,
                    "l4_O": regs.l4.O,
                    "l4_E": regs.l4.E,
                })
        return out

    def generate_full_coverage(self) -> List[Dict[str, Any]]:
        """Ensure every byte from every vertex charge is represented."""
        out = []
        for b in range(256):
            self.four.reset()
            self.four.ingest_byte(b)
            regs = self.four.regs
            m12 = physics.expand_intron_to_mask12(physics.intron(b))
            v = physics.vertex_charge(m12)
            out.append({
                "input_ids": [b],
                "l1_state8": regs.l1_state8,
                "l2_state16": regs.l2_state16,
                "l3_state24": regs.l3_state24,
                "l4_O": regs.l4.O,
                "l4_E": regs.l4.E,
                "l4_parity": regs.l4.parity,
                "vertex": v,
            })
        return out

    def save_curriculum(self, output_dir: Path, scale: int = 1) -> None:
        """
        V2 curriculum: physics-grounded structured policies + some random.
        Phase 1 goal: learn physics-useful stochastic structure, not language.
        Types appended per-walk for robustness.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        walks: List[Dict[str, Any]] = []
        types: List[int] = []

        def add(type_id: int, seqs: List[Dict[str, Any]]) -> None:
            walks.extend(seqs)
            types.extend([type_id] * len(seqs))

        if scale >= 10:
            L = 128
            add(0, self.generate_repeat_patterns(3000, L, seed=1))
            add(1, self.generate_family_locked(3000, L, seed=2))
            add(2, self.generate_micro_locked(3000, L, seed=3))
            add(3, self.generate_vertex_locked(2000, L, seed=4))
            add(4, self.generate_separator_patterns(2000, L, seed=5))
            add(5, self.generate_horizon_walks(2000, L, seed=6))
            add(6, self.generate_closure_walks(2000, seq_len=L, seed=7))
            add(7, self.generate_random_walks(2000, L, seed=8))
            add(8, self.generate_full_coverage())
            desc = f"V2 full: {len(walks)} sequences (structured-heavy)"
        else:
            L = 64
            add(0, self.generate_repeat_patterns(400, L, seed=1))
            add(1, self.generate_family_locked(400, L, seed=2))
            add(2, self.generate_micro_locked(400, L, seed=3))
            add(3, self.generate_vertex_locked(200, L, seed=4))
            add(4, self.generate_separator_patterns(200, L, seed=5))
            add(5, self.generate_horizon_walks(200, L, seed=6))
            add(6, self.generate_closure_walks(200, seq_len=L, seed=7))
            add(7, self.generate_random_walks(400, L, seed=8))
            add(8, self.generate_full_coverage())
            desc = f"V2 debug: {len(walks)} sequences"

        meta = {"num_sequences": len(walks), "description": desc, "version": 2, "scale": scale}
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        with open(output_dir / "trajectories.bin", "wb") as f:
            for w in walks:
                ids = bytes(w["input_ids"])
                f.write(len(ids).to_bytes(4, "little"))
                f.write(ids)
        with open(output_dir / "types.bin", "wb") as f:
            f.write(bytes(types))

    def load_curriculum(self, input_dir: Path) -> "CurriculumDataset":
        """Load as a standard PyTorch Dataset."""
        return CurriculumDataset(Path(input_dir))


CURRICULUM_TYPE_NAMES = [
    "repeat", "family_locked", "micro_locked", "vertex_locked",
    "separator", "horizon", "closure", "random", "full_coverage",
]


class CurriculumDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.sequences = []
        self.sequence_types = []
        path = self.data_dir / "trajectories.bin"
        if path.exists():
            with open(path, "rb") as f:
                idx = 0
                while True:
                    sz_b = f.read(4)
                    if not sz_b:
                        break
                    sz = int.from_bytes(sz_b, "little")
                    ids = list(f.read(sz))
                    if len(ids) == sz:
                        self.sequences.append(ids)
                        idx += 1
        types_path = self.data_dir / "types.bin"
        if types_path.exists() and len(self.sequences) > 0:
            with open(types_path, "rb") as f:
                raw = f.read()
            self.sequence_types = list(raw[: len(self.sequences)])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        out: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(seq, dtype=torch.long),
        }
        if self.sequence_types:
            out["type_id"] = torch.tensor(self.sequence_types[idx], dtype=torch.long)
        return out
