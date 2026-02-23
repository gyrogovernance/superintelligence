"""
Module 1: Observer — kernel runs alongside Bolmo, records observables, changes nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common import token_to_byte_and_fused
from src.router.constants import (
    ARCHETYPE_A12,
    LAYER_MASK_12,
    mask12_for_byte,
    popcount,
    unpack_state,
    vertex_charge_from_mask,
)
from src.router.kernel import RouterKernel


@dataclass
class ByteObservation:
    """One byte's kernel + Bolmo observables."""
    position: int
    byte_value: int
    intron: int
    micro: int
    family: int
    mask12: int
    mask_weight: int
    state_index: int
    a12: int
    b12: int
    horizon_index: int
    horizon_distance: int
    vertex_charge: int
    phase: int
    token_id: int = -1
    is_fused: bool = False


@dataclass
class ObservationLog:
    """Full sequence of observations."""
    observations: list[ByteObservation] = field(default_factory=list)

    def bytes_list(self) -> list[int]:
        return [o.byte_value for o in self.observations]

    def boundaries(self) -> list[int]:
        return [i for i, o in enumerate(self.observations) if o.is_fused]

    def boundary_rate(self) -> float:
        if not self.observations:
            return 0.0
        return len(self.boundaries()) / len(self.observations)

    def horizon_coverage(self) -> int:
        return len(set(o.horizon_index for o in self.observations))

    def vertex_distribution(self) -> dict[int, int]:
        return dict(sorted(Counter(o.vertex_charge for o in self.observations).items()))

    def phase_distribution(self) -> dict[int, int]:
        return dict(sorted(Counter(o.phase for o in self.observations).items()))

    def family_distribution(self) -> dict[int, int]:
        return dict(sorted(Counter(o.family for o in self.observations).items()))

    def mean_mask_weight(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([o.mask_weight for o in self.observations]))

    def mean_horizon_distance(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([o.horizon_distance for o in self.observations]))

    def code_distances(self) -> list[int]:
        dists = []
        for i in range(1, len(self.observations)):
            d = popcount(self.observations[i].mask12 ^ self.observations[i - 1].mask12)
            dists.append(d)
        return dists

    def phase_transitions(self) -> list[int]:
        trans = []
        for i in range(1, len(self.observations)):
            dp = abs(self.observations[i].phase - self.observations[i - 1].phase) % 4
            trans.append(dp)
        return trans

    def vertex_transitions(self) -> list[bool]:
        trans = []
        for i in range(1, len(self.observations)):
            trans.append(self.observations[i].vertex_charge != self.observations[i - 1].vertex_charge)
        return trans


class KernelObserver:
    """Runs RouterKernel alongside Bolmo, records everything, changes nothing."""

    def __init__(self, atlas_dir: Path):
        self.kernel = RouterKernel(atlas_dir=atlas_dir)
        self.log = ObservationLog()
        self._pos = 0
        self._mask_to_byte: dict[int, int] = {}
        for b in range(256):
            self._mask_to_byte[mask12_for_byte(b)] = b

    def reset(self) -> None:
        self.kernel.reset()
        self.log = ObservationLog()
        self._pos = 0

    def observe_byte(
        self,
        byte_value: int,
        token_id: int = -1,
        is_fused: bool = False,
    ) -> ByteObservation:
        bv = int(byte_value) & 0xFF
        intron = bv ^ 0xAA
        micro = intron & 0x3F
        family = (intron >> 6) & 0x3
        mask = mask12_for_byte(bv)
        mask_weight = popcount(mask)
        vertex = vertex_charge_from_mask(mask)

        self.kernel.step_byte(bv)

        si = int(self.kernel.state_index.flat[0])
        s24 = int(self.kernel.ontology[si])
        a12, b12 = unpack_state(s24)
        m = a12 ^ ARCHETYPE_A12
        horizon_index = self._mask_to_byte.get(m, -1)
        horizon_distance = popcount(a12 ^ (b12 ^ LAYER_MASK_12))
        phase = int(self.kernel.phase[si, bv])

        obs = ByteObservation(
            position=self._pos,
            byte_value=bv,
            intron=intron,
            micro=micro,
            family=family,
            mask12=mask,
            mask_weight=mask_weight,
            state_index=si,
            a12=a12,
            b12=b12,
            horizon_index=horizon_index,
            horizon_distance=horizon_distance,
            vertex_charge=vertex,
            phase=phase,
            token_id=token_id,
            is_fused=is_fused,
        )
        self.log.observations.append(obs)
        self._pos += 1
        return obs

    def observe_token_sequence(
        self, token_ids: list[int], token_offset: int
    ) -> list[ByteObservation]:
        results = []
        for tid in token_ids:
            b, fused = token_to_byte_and_fused(tid, token_offset)
            if b is not None:
                obs = self.observe_byte(b, token_id=tid, is_fused=fused)
                results.append(obs)
        return results


def print_observation_summary(log: ObservationLog, label: str = "", max_show: int = 40) -> None:
    obs = log.observations
    print(f"{label}Trajectory: {len(obs)} bytes")
    if not obs:
        return

    print(f"  Boundaries: {len(log.boundaries())} ({100*log.boundary_rate():.1f}%)")
    print(f"  Horizon coverage: {log.horizon_coverage()}/256")
    print(f"  Vertex dist: {log.vertex_distribution()}")
    print(f"  Phase dist: {log.phase_distribution()}")
    print(f"  Family dist: {log.family_distribution()}")
    print(f"  Mean mask weight: {log.mean_mask_weight():.2f}")
    print(f"  Mean horizon dist: {log.mean_horizon_distance():.2f}")

    if len(obs) > 1:
        cd = log.code_distances()
        pt = log.phase_transitions()
        vt = log.vertex_transitions()
        print(f"  Mean code distance: {np.mean(cd):.2f}")
        print(f"  Phase transition dist: {dict(sorted(Counter(pt).items()))}")
        print(f"  Vertex change rate: {sum(vt)/len(vt):.3f}")

        if len(log.boundaries()) > 0 and len(log.boundaries()) < len(obs) - 1:
            bnd_set = set(log.boundaries())
            pt_at_bnd = [pt[i - 1] for i in bnd_set if 0 < i <= len(pt)]
            pt_at_non = [pt[i - 1] for i in range(1, len(obs)) if i not in bnd_set and i - 1 < len(pt)]
            if pt_at_bnd and pt_at_non:
                print(f"  Phase trans at boundary: mean={np.mean(pt_at_bnd):.3f} (n={len(pt_at_bnd)})")
                print(f"  Phase trans at non-bnd:  mean={np.mean(pt_at_non):.3f} (n={len(pt_at_non)})")

    n = min(max_show, len(obs))
    print(f"  First {n} bytes:")
    print(f"  {'pos':>4} {'byte':>4} {'chr':>3} {'mic':>3} {'fam':>3} {'mw':>2} {'h':>3} {'HD':>2} {'chi':>3} {'ph':>2} {'fsd':>3}")
    for o in obs[:n]:
        ch = chr(o.byte_value) if 32 <= o.byte_value < 127 else '.'
        f = 'F' if o.is_fused else '.'
        print(f"  {o.position:>4} {o.byte_value:>4} {ch:>3} {o.micro:>3} {o.family:>3} {o.mask_weight:>2} {o.horizon_index:>3} {o.horizon_distance:>2} {o.vertex_charge:>3} {o.phase:>2} {f:>3}")
