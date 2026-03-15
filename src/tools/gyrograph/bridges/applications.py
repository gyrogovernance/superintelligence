from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.tools.gyrograph import GyroGraph
from src.tools.gyrograph.profiles import ResonanceProfile
from src.tools.gyrograph import ops as gg_ops
from src.tools.gyrograph.serializers import pack_word4


# Fixed mapping of structural runtime events to exact kernel words.
# Similar runtime operations map to the same word or same q-class.
# No hashing: cryptographic hashes destroy topology via avalanche effect.
_EVENT_VOCABULARY = {
    # Hot-loop family (0xAA base)
    "loop_enter": pack_word4(0xAA, 0x10, 0x20, 0x30),
    "loop_body": pack_word4(0xAA, 0x10, 0x20, 0x31),
    "loop_jump": pack_word4(0xAA, 0x10, 0x20, 0x32),
    # GC family (0x54 base)
    "gc_start": pack_word4(0x54, 0x00, 0xF0, 0x00),
    "gc_mark": pack_word4(0x54, 0x00, 0xF0, 0x01),
    "gc_scan": pack_word4(0x54, 0x00, 0xF0, 0x02),
    "gc_sweep": pack_word4(0x54, 0x00, 0xF0, 0x03),
    "gc_finalize": pack_word4(0x54, 0x00, 0xF0, 0x04),
    "gc_idle": pack_word4(0x54, 0x00, 0xF0, 0x05),
    # Contention family (0xD5 base, C-gates style)
    "wait_lock": pack_word4(0xD5, 0x2B, 0xD5, 0x2B),
    "wake": pack_word4(0xD5, 0x2B, 0xD5, 0x2C),
    "retry": pack_word4(0xD5, 0x2B, 0xD5, 0x2D),
    "yield": pack_word4(0xD5, 0x2B, 0xD5, 0x2E),
    "spin": pack_word4(0xD5, 0x2B, 0xD5, 0x2F),
    "sleep": pack_word4(0xD5, 0x2B, 0xD5, 0x30),
}


@dataclass(frozen=True)
class ApplicationEvent:
    entity_id: str
    event_type: str
    index: int
    role: str = "main"
    region: str = ""
    thread_id: int = 0
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ApplicationDecision:
    entity_id: str
    role: str
    cell_id: int
    hot_loop_score: float
    contention_score: float
    suggested_action: str
    resonance_key: int
    current_resonance: int
    shell: int
    chi6: int
    spectral_norm: float
    spectral_peak_ratio: float
    chi_support_ratio: float
    chi_peak_ratio: float
    shell_entropy: float


class ApplicationsBridge:
    """
    Applications-domain GyroGraph bridge.

    This bridge owns:
    - deterministic runtime-event -> word4 serialization
    - entity_id / role -> cell_id mapping
    - actuator-oriented profiling for the two core spec scenarios:
      A1. hot-loop specialization
      A2. lock/contention style instability
    """

    HOT_LOOP_PATTERN = ("loop_enter", "loop_body", "loop_body", "loop_jump")
    GC_PATTERN = ("gc_start", "gc_mark", "gc_scan", "gc_sweep", "gc_finalize", "gc_idle")
    CONTENTION_PATTERN = (
        "wait_lock",
        "wake",
        "retry",
        "yield",
        "spin",
        "sleep",
        "wake",
        "retry",
    )

    def __init__(
        self,
        *,
        graph: GyroGraph | None = None,
        cell_capacity: int = 1024,
        profile: ResonanceProfile = ResonanceProfile.CHIRALITY,
        use_native_hotpath: bool = True,
        use_opencl_hotpath: bool = True,
        opencl_min_batch: int = 128,
        enable_ingest_log: bool = False,
        ingest_log_path: str | None = None,
    ) -> None:
        self.graph = graph or GyroGraph(
            cell_capacity=cell_capacity,
            profile=profile,
            use_native_hotpath=use_native_hotpath,
            use_opencl_hotpath=use_opencl_hotpath,
            opencl_min_batch=opencl_min_batch,
            enable_ingest_log=enable_ingest_log,
            ingest_log_path=ingest_log_path,
        )
        self._entity_roles: dict[str, dict[str, int]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Runtime capabilities
    # ------------------------------------------------------------------

    def runtime_capabilities(self) -> dict[str, Any]:
        return {
            "gyrograph_native_available": gg_ops.native_available(),
            "gyrograph_opencl_available": gg_ops.opencl_available(),
            "graph_native_hotpath_enabled": bool(
                getattr(self.graph, "_use_native_hotpath", False)
            ),
            "graph_opencl_hotpath_enabled": bool(
                getattr(self.graph, "_use_opencl_hotpath", False)
            ),
            "graph_profile": self.graph.profile.name,
            "graph_capacity": self.graph.capacity,
            "graph_active_cells": self.graph.active_cell_count,
        }

    # ------------------------------------------------------------------
    # Entity / role mapping
    # ------------------------------------------------------------------

    def ensure_role(self, entity_id: str, role: str = "main") -> int:
        entity = str(entity_id)
        role_s = str(role)
        if role_s in self._entity_roles[entity]:
            return self._entity_roles[entity][role_s]
        [cid] = self.graph.allocate_cells(1)
        self._entity_roles[entity][role_s] = cid
        return cid

    def entity_cell(self, entity_id: str, role: str = "main") -> int:
        entity = str(entity_id)
        role_s = str(role)
        if entity not in self._entity_roles or role_s not in self._entity_roles[entity]:
            raise KeyError(f"No cell bound for entity={entity!r}, role={role_s!r}")
        return self._entity_roles[entity][role_s]

    def entity_cells(self, entity_id: str) -> dict[str, int]:
        entity = str(entity_id)
        return dict(self._entity_roles.get(entity, {}))

    # ------------------------------------------------------------------
    # Serializer
    # ------------------------------------------------------------------

    @staticmethod
    def event_to_word4(
        event_type: str,
        index: int,
        *,
        role: str = "main",
        region: str = "",
        thread_id: int = 0,
        metadata: dict[str, Any] | None = None,
        phase_mod: int | None = 8,
    ) -> bytes:
        """
        Structure-preserving applications-domain serializer.

        Maps recognized runtime event types to stable, predetermined 4-byte words
        via categorical lookup. Similar events (e.g. loop_body, loop_jump) share
        the same q-class. Unknown events fall back to identity-like word.
        """
        event_s = str(event_type)
        return _EVENT_VOCABULARY.get(event_s, b"\xAA\xAA\xAA\xAA")

    def events_to_packets(
        self,
        events: Iterable[ApplicationEvent],
        *,
        phase_mod: int | None = 8,
    ) -> list[tuple[int, bytes]]:
        packets: list[tuple[int, bytes]] = []
        for ev in events:
            cid = self.ensure_role(ev.entity_id, ev.role)
            metadata = dict(ev.metadata)
            word4 = self.event_to_word4(
                ev.event_type,
                ev.index,
                role=ev.role,
                region=ev.region,
                thread_id=ev.thread_id,
                metadata=metadata,
                phase_mod=phase_mod,
            )
            packets.append((cid, word4))
        return packets

    # ------------------------------------------------------------------
    # Ingest surfaces
    # ------------------------------------------------------------------

    def ingest_event(
        self,
        event: ApplicationEvent,
        *,
        phase_mod: int | None = 8,
    ) -> int:
        cid = self.ensure_role(event.entity_id, event.role)
        word4 = self.event_to_word4(
            event.event_type,
            event.index,
            role=event.role,
            region=event.region,
            thread_id=event.thread_id,
            metadata=dict(event.metadata),
            phase_mod=phase_mod,
        )
        self.graph.ingest([(cid, word4)])
        return cid

    def ingest_events(
        self,
        events: Iterable[ApplicationEvent],
        *,
        phase_mod: int | None = 8,
    ) -> list[int]:
        packets = self.events_to_packets(events, phase_mod=phase_mod)
        self.graph.ingest(packets)
        return [cid for cid, _ in packets]

    # ------------------------------------------------------------------
    # Scenario helpers from the spec
    # ------------------------------------------------------------------

    def feed_hot_loop(
        self,
        entity_id: str,
        *,
        iterations: int,
        role: str = "main",
        region: str = "hot_loop",
        thread_id: int = 0,
    ) -> int:
        events = [
            ApplicationEvent(
                entity_id=str(entity_id),
                event_type=self.HOT_LOOP_PATTERN[i % len(self.HOT_LOOP_PATTERN)],
                index=i,
                role=role,
                region=region,
                thread_id=thread_id,
                metadata=(("scenario", "hot_loop"),),
            )
            for i in range(int(iterations))
        ]
        self.ingest_events(events, phase_mod=len(self.HOT_LOOP_PATTERN))
        return self.entity_cell(entity_id, role)

    def feed_gc_cycle(
        self,
        entity_id: str,
        *,
        cycles: int,
        role: str = "main",
        region: str = "gc",
        thread_id: int = 0,
    ) -> int:
        pattern = self.GC_PATTERN
        total = int(cycles) * len(pattern)
        events = [
            ApplicationEvent(
                entity_id=str(entity_id),
                event_type=pattern[i % len(pattern)],
                index=i,
                role=role,
                region=region,
                thread_id=thread_id,
                metadata=(("scenario", "gc_cycle"),),
            )
            for i in range(total)
        ]
        self.ingest_events(events, phase_mod=len(pattern))
        return self.entity_cell(entity_id, role)

    def feed_lock_contention(
        self,
        entity_id: str,
        *,
        cycles: int,
        role: str = "lock",
        region: str = "lock_contention",
        thread_id: int = 0,
    ) -> int:
        pattern = self.CONTENTION_PATTERN
        total = int(cycles) * len(pattern)
        events = [
            ApplicationEvent(
                entity_id=str(entity_id),
                event_type=pattern[i % len(pattern)],
                index=i,
                role=role,
                region=region,
                thread_id=thread_id,
                metadata=(("scenario", "lock_contention"),),
            )
            for i in range(total)
        ]
        self.ingest_events(events, phase_mod=len(pattern))
        return self.entity_cell(entity_id, role)

    # ------------------------------------------------------------------
    # SLCP and entity views
    # ------------------------------------------------------------------

    def emit_entity_slcp(self, entity_id: str, role: str = "main"):
        cid = self.entity_cell(entity_id, role)
        return self.graph.emit_slcp([cid])[0]

    def emit_entity_records(self, entity_id: str) -> dict[str, Any]:
        role_map = self.entity_cells(entity_id)
        out: dict[str, Any] = {}
        for role, cid in role_map.items():
            out[role] = self.graph.emit_slcp([cid])[0]
        return out

    # ------------------------------------------------------------------
    # Actuator scoring
    # ------------------------------------------------------------------

    def _cell_statistics(self, cell_id: int) -> dict[str, float]:
        cid = int(cell_id)
        valid = int(self.graph._chi_valid_len[cid])

        if valid <= 0:
            return {
                "valid_len": 0.0,
                "chi_support_ratio": 0.0,
                "chi_peak_ratio": 0.0,
                "shell_entropy": 0.0,
                "spectral_norm": 0.0,
                "spectral_peak_ratio": 0.0,
            }

        chi_hist = self.graph._chi_hist64[cid].astype(np.float64, copy=False)
        shell_hist = self.graph._shell_hist7[cid].astype(np.float64, copy=False)
        spec = self.graph.emit_slcp([cid])[0].spectral64.astype(np.float64, copy=False)

        chi_support_ratio = float(np.count_nonzero(chi_hist) / valid)
        chi_peak_ratio = float(chi_hist.max() / valid)

        shell_total = shell_hist.sum()
        if shell_total > 0:
            p = shell_hist / shell_total
            p = p[p > 0]
            shell_entropy = float(-(p * np.log2(p)).sum() / math.log2(7.0))
        else:
            shell_entropy = 0.0

        spectral_norm = float(np.linalg.norm(spec))
        if spectral_norm > 0:
            spectral_peak_ratio = float(np.max(np.abs(spec)) / spectral_norm)
        else:
            spectral_peak_ratio = 0.0

        return {
            "valid_len": float(valid),
            "chi_support_ratio": chi_support_ratio,
            "chi_peak_ratio": chi_peak_ratio,
            "shell_entropy": shell_entropy,
            "spectral_norm": spectral_norm,
            "spectral_peak_ratio": spectral_peak_ratio,
        }

    def hot_loop_score(self, entity_id: str, role: str = "main") -> float:
        cid = self.entity_cell(entity_id, role)
        s = self._cell_statistics(cid)
        # Hot loops have low support (few states), low entropy, and strong spectral peaks.
        return float(
            0.50 * (1.0 - s["chi_support_ratio"])
            + 0.30 * (1.0 - (s["shell_entropy"] / 1.0))
            + 0.20 * s["spectral_peak_ratio"]
        )

    def contention_score(self, entity_id: str, role: str = "main") -> float:
        cid = self.entity_cell(entity_id, role)
        s = self._cell_statistics(cid)
        # Contention has high support (wandering state), high entropy, flat spectra.
        return float(
            0.50 * s["chi_support_ratio"]
            + 0.30 * (s["shell_entropy"] / 1.0)
            + 0.20 * (1.0 - s["spectral_peak_ratio"])
        )

    def profile_entity(self, entity_id: str, role: str = "main") -> ApplicationDecision:
        cid = self.entity_cell(entity_id, role)
        rec = self.emit_entity_slcp(entity_id, role)
        stats = self._cell_statistics(cid)

        hot = self.hot_loop_score(entity_id, role)
        cont = self.contention_score(entity_id, role)

        if hot >= 0.58 and hot > cont + 0.03:
            action = "specialize_hot_loop"
        elif cont >= 0.58 and cont > hot + 0.03:
            action = "mitigate_contention"
        else:
            action = "observe"

        return ApplicationDecision(
            entity_id=str(entity_id),
            role=str(role),
            cell_id=cid,
            hot_loop_score=hot,
            contention_score=cont,
            suggested_action=action,
            resonance_key=int(rec.resonance_key),
            current_resonance=int(rec.current_resonance),
            shell=int(rec.shell),
            chi6=int(rec.chi6),
            spectral_norm=float(stats["spectral_norm"]),
            spectral_peak_ratio=float(stats["spectral_peak_ratio"]),
            chi_support_ratio=float(stats["chi_support_ratio"]),
            chi_peak_ratio=float(stats["chi_peak_ratio"]),
            shell_entropy=float(stats["shell_entropy"]),
        )

    def compare_entities(
        self,
        entity_a: str,
        entity_b: str,
        *,
        role_a: str = "main",
        role_b: str = "main",
    ) -> dict[str, Any]:
        rec_a = self.emit_entity_slcp(entity_a, role_a)
        rec_b = self.emit_entity_slcp(entity_b, role_b)

        spectral_l2 = float(
            np.linalg.norm(
                rec_a.spectral64.astype(np.float64) - rec_b.spectral64.astype(np.float64)
            )
        )
        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "cell_a": self.entity_cell(entity_a, role_a),
            "cell_b": self.entity_cell(entity_b, role_b),
            "same_resonance_bucket": bool(rec_a.resonance_key == rec_b.resonance_key),
            "chirality_distance": self.graph.chirality_distance_between_cells(
                self.entity_cell(entity_a, role_a),
                self.entity_cell(entity_b, role_b),
            ),
            "spectral_l2": spectral_l2,
            "shell_a": int(rec_a.shell),
            "shell_b": int(rec_b.shell),
        }