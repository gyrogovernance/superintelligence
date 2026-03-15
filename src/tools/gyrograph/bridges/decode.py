from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch

from src.api import q_word6, shadow_partner_byte
from src.tools.gyrograph import GyroGraph
from src.tools.gyrograph.profiles import ResonanceProfile, shell_from_omega12
from src.tools.gyrograph.serializers import pack_word4
from .bolmo_config import (
    BolmoDecodeBridgeConfig,
    canonical_byte_from_token_id,
    build_metric_byte,
    strip_boundary_phase,
)


_Q_LUT_256 = np.asarray([q_word6(b) for b in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class PairedContentMetrics:
    support_ratio: float
    peak_ratio: float
    gauge_abs: float
    q_support_mask: int
    dominant_q6: int
    support_count: int
    raw_support_ratio: float
    raw_support_count: int


@dataclass(frozen=True)
class PairedStepRecord:
    step: int
    support_ratio: float
    peak_ratio: float
    gauge_abs: float
    dominant_q6: int
    support_count: int
    raw_support_ratio: float
    raw_support_count: int
    selected_token_id: int | None
    selected_canonical_byte: int | None
    selected_boundary: bool | None


@dataclass(frozen=True)
class PatchRecord:
    patch_index: int
    bytes_payload: bytes
    length: int
    ended_by_boundary: bool


@dataclass
class _StreamState:
    stream_id: str
    network_cell: int
    database_cell: int
    application_cell: int
    network_buffer: list[int] = field(default_factory=list)
    database_buffer: list[int] = field(default_factory=list)
    application_buffer: list[int] = field(default_factory=list)
    step: int = 0
    pending_metrics: PairedContentMetrics | None = None
    current_patch_bytes: list[int] = field(default_factory=list)
    patch_records: list[PatchRecord] = field(default_factory=list)
    step_records: list[PairedStepRecord] = field(default_factory=list)
    boundary_emit_count: int = 0
    gauge_flip_count: int = 0
    previous_boundary: bool | None = None

    def clear(self) -> None:
        self.network_buffer.clear()
        self.database_buffer.clear()
        self.application_buffer.clear()
        self.step = 0
        self.pending_metrics = None
        self.current_patch_bytes.clear()
        self.patch_records.clear()
        self.step_records.clear()
        self.boundary_emit_count = 0
        self.gauge_flip_count = 0
        self.previous_boundary = None


@dataclass(frozen=True)
class BolmoDecodeReport:
    stream_id: str
    network: dict[str, Any]
    database: dict[str, Any]
    application: dict[str, Any]
    patch_geometry: dict[str, Any]


class GyroGraphBolmoDecodeBridge:
    """
    Decode-only Bolmo climate bridge.

    Responsibilities:
      - factor Bolmo decode into content and boundary/gauge channels
      - pair normal/fused logits before selection
      - serialize model-native climate bytes into 3 GyroGraph domain cells
      - read shell/gauge/order-parameter observables from the occupied QuBEC
    """

    def __init__(
        self,
        *,
        graph: GyroGraph | None = None,
        config: BolmoDecodeBridgeConfig | None = None,
        cell_capacity: int = 4096,
        profile: ResonanceProfile = ResonanceProfile.CHIRALITY,
        use_native_hotpath: bool = True,
        use_opencl_hotpath: bool = True,
        opencl_min_batch: int = 128,
        enable_ingest_log: bool = False,
        ingest_log_path: str | None = None,
    ) -> None:
        self.config = config or BolmoDecodeBridgeConfig()
        self.graph = graph or GyroGraph(
            cell_capacity=cell_capacity,
            profile=profile,
            use_native_hotpath=use_native_hotpath,
            use_opencl_hotpath=use_opencl_hotpath,
            opencl_min_batch=opencl_min_batch,
            enable_ingest_log=enable_ingest_log,
            ingest_log_path=ingest_log_path,
        )

        self._streams: dict[str, _StreamState] = {}
        self._active_stream_ids: list[str] = []

        self._attached_model: Any = None
        self._prev_boundary_hook: Any = None
        self._prev_token_hook: Any = None
        self._prev_select_hook: Any = None

        self._q_lut_t = torch.tensor(_Q_LUT_256.tolist(), dtype=torch.long)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _decode_target(self, model: Any) -> Any:
        return getattr(model, "base_model", model)

    def _ensure_stream(self, stream_id: str) -> _StreamState:
        if stream_id in self._streams:
            return self._streams[stream_id]

        c_net, c_db, c_app = self.graph.allocate_cells(3)
        st = _StreamState(
            stream_id=stream_id,
            network_cell=c_net,
            database_cell=c_db,
            application_cell=c_app,
        )
        self._streams[stream_id] = st
        return st

    def start(self, batch_size: int, stream_ids: list[str] | None = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if stream_ids is None:
            stream_ids = [
                f"{self.config.stream_prefix}:{i}"
                for i in range(batch_size)
            ]
        if len(stream_ids) != batch_size:
            raise ValueError(
                f"Expected {batch_size} stream_ids, got {len(stream_ids)}"
            )

        self._active_stream_ids = list(stream_ids)

        for sid in self._active_stream_ids:
            st = self._ensure_stream(sid)
            self.graph.seed_rest(
                [st.network_cell, st.database_cell, st.application_cell]
            )
            st.clear()

    def attach(
        self,
        model: Any,
        *,
        batch_size: int,
        stream_ids: list[str] | None = None,
    ) -> None:
        target = self._decode_target(model)

        self._prev_boundary_hook = getattr(target, "_decode_boundary_hook", None)
        self._prev_token_hook = getattr(target, "_decode_token_hook", None)
        self._prev_select_hook = getattr(target, "_decode_select_hook", None)

        target._decode_boundary_hook = self.boundary_hook
        target._decode_token_hook = self.token_hook
        target._decode_select_hook = self.select_hook

        self._attached_model = target
        self.start(batch_size=batch_size, stream_ids=stream_ids)

    def detach(self) -> None:
        if self._attached_model is not None:
            self._attached_model._decode_boundary_hook = self._prev_boundary_hook
            self._attached_model._decode_token_hook = self._prev_token_hook
            self._attached_model._decode_select_hook = self._prev_select_hook

        self._attached_model = None
        self._prev_boundary_hook = None
        self._prev_token_hook = None
        self._prev_select_hook = None
        self._active_stream_ids = []

    @contextmanager
    def session(
        self,
        model: Any,
        *,
        batch_size: int,
        stream_ids: list[str] | None = None,
    ) -> Iterator["GyroGraphBolmoDecodeBridge"]:
        self.attach(model, batch_size=batch_size, stream_ids=stream_ids)
        try:
            yield self
        finally:
            self.detach()

    def generate(
        self,
        model: Any,
        input_ids: torch.Tensor,
        *,
        stream_ids: list[str] | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        with self.session(
            model,
            batch_size=int(input_ids.shape[0]),
            stream_ids=stream_ids,
        ):
            return model.generate(input_ids, **generate_kwargs)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def runtime_capabilities(self) -> dict[str, Any]:
        return {
            "graph_native_hotpath_enabled": self.graph.native_hotpath_enabled,
            "graph_opencl_hotpath_enabled": self.graph.opencl_hotpath_enabled,
            "graph_ingest_log_enabled": self.graph.ingest_log_enabled,
            "graph_profile": self.graph.profile.name,
            "graph_capacity": self.graph.capacity,
            "graph_active_cells": self.graph.active_cell_count,
            "token_layout": {
                "offset": self.config.token_layout.offset,
                "special_count": self.config.token_layout.special_count,
                "boundary_offset": self.config.token_layout.boundary_offset,
            },
        }

    # ------------------------------------------------------------------
    # Pairing / climate metrics
    # ------------------------------------------------------------------

    def _pair_row(self, row: torch.Tensor) -> PairedContentMetrics:
        cfg = self.config.token_layout

        normal = row[cfg.normal_byte_low : cfg.normal_byte_high + 1]
        fused = row[
            cfg.normal_byte_low + cfg.boundary_offset :
            cfg.normal_byte_high + cfg.boundary_offset + 1
        ]

        # Raw 512-way byte-phase competition
        raw_logits = torch.cat([normal, fused], dim=0)
        raw_probs = torch.softmax(raw_logits, dim=-1)
        raw_floor = 0.5 * float(self.config.content_probability_floor)
        raw_support_mask = raw_probs >= raw_floor
        raw_support_count = int(raw_support_mask.sum().item())
        raw_support_ratio = float(raw_support_count / 512.0)

        # Paired quotient competition over 256 content classes
        content_mass = torch.logaddexp(normal, fused)
        content_probs = torch.softmax(content_mass, dim=-1)

        support_mask = content_probs >= self.config.content_probability_floor
        support_count = int(support_mask.sum().item())
        support_ratio = float(support_count / 256.0)
        peak_ratio = float(content_probs.max().item())

        gauge_delta = fused - normal
        gauge_abs = float((content_probs * gauge_delta.abs()).sum().item())

        q_ids = self._q_lut_t.to(row.device)
        q_mass = torch.bincount(
            q_ids,
            weights=content_probs.to(torch.float32),
            minlength=64,
        )
        dominant_q6 = int(torch.argmax(q_mass).item())

        q_support_mask = 0
        for q in q_ids[support_mask].tolist():
            q_support_mask |= int(q)

        return PairedContentMetrics(
            support_ratio=support_ratio,
            peak_ratio=peak_ratio,
            gauge_abs=gauge_abs,
            q_support_mask=q_support_mask & 0x3F,
            dominant_q6=dominant_q6 & 0x3F,
            support_count=support_count,
            raw_support_ratio=raw_support_ratio,
            raw_support_count=raw_support_count,
        )

    def _application_phase_scale(self, stream: _StreamState) -> float:
        shell = shell_from_omega12(int(self.graph._omega12[stream.application_cell]))
        phase_scale = 1.0 - self.config.application_phase_damping * (shell / 6.0)
        return float(
            max(self.config.min_phase_scale, min(self.config.max_phase_scale, phase_scale))
        )

    def _apply_application_gauge_control(
        self,
        row: torch.Tensor,
        stream: _StreamState,
    ) -> None:
        cfg = self.config.token_layout

        normal_slice = slice(cfg.normal_byte_low, cfg.normal_byte_high + 1)
        fused_slice = slice(
            cfg.normal_byte_low + cfg.boundary_offset,
            cfg.normal_byte_high + cfg.boundary_offset + 1,
        )

        normal = row[normal_slice].clone()
        fused = row[fused_slice].clone()

        mean = 0.5 * (normal + fused)
        delta = 0.5 * (fused - normal)

        phase_scale = self._application_phase_scale(stream)

        row[normal_slice] = mean - (phase_scale * delta)
        row[fused_slice] = mean + (phase_scale * delta)

    def _apply_database_sector_control(
        self,
        row: torch.Tensor,
        metrics: PairedContentMetrics,
    ) -> None:
        if metrics.support_ratio <= self.config.database_support_target:
            return

        cfg = self.config.token_layout
        bonus = self.config.database_sector_bonus * (
            metrics.support_ratio - self.config.database_support_target
        )

        q_mask = (_Q_LUT_256 == metrics.dominant_q6)
        q_mask_t = torch.tensor(q_mask.tolist(), dtype=torch.bool, device=row.device)

        normal_slice = slice(cfg.normal_byte_low, cfg.normal_byte_high + 1)
        fused_slice = slice(
            cfg.normal_byte_low + cfg.boundary_offset,
            cfg.normal_byte_high + cfg.boundary_offset + 1,
        )

        normal = row[normal_slice].clone()
        fused = row[fused_slice].clone()

        normal[q_mask_t] += bonus
        fused[q_mask_t] += bonus

        row[normal_slice] = normal
        row[fused_slice] = fused

    # ------------------------------------------------------------------
    # Model hook surfaces
    # ------------------------------------------------------------------

    def boundary_hook(
        self,
        next_token_logits: torch.Tensor,
        last_bytes: list[int],
        boundary_offset: int,
    ) -> torch.Tensor:
        if next_token_logits.ndim != 3:
            raise ValueError(
                f"Expected next_token_logits [B,1,V], got shape {tuple(next_token_logits.shape)}"
            )
        if len(self._active_stream_ids) != next_token_logits.shape[0]:
            raise RuntimeError(
                "Decode bridge session batch_size does not match logits batch"
            )

        out = next_token_logits.clone()
        rows = out[:, -1, :]

        mode = getattr(self.config, "control_mode", "observe") or "observe"
        for i, sid in enumerate(self._active_stream_ids):
            st = self._streams[sid]
            metrics = self._pair_row(rows[i])
            st.pending_metrics = metrics

            if mode == "gauge_damp" or mode == "full":
                self._apply_application_gauge_control(rows[i], st)
            if mode == "sector_shape" or mode == "full":
                self._apply_database_sector_control(rows[i], metrics)

        out[:, -1, :] = rows
        return out

    def select_hook(
        self,
        next_token_scores: torch.Tensor,
        boundary_offset: int,
    ) -> torch.Tensor:
        if next_token_scores.ndim == 3:
            next_token_scores = next_token_scores[:, -1, :]
        if next_token_scores.ndim != 2:
            raise ValueError(
                f"Expected next_token_scores [B,V], got shape {tuple(next_token_scores.shape)}"
            )
        if next_token_scores.shape[0] != len(self._active_stream_ids):
            raise RuntimeError(
                "Decode bridge session batch_size does not match score batch"
            )

        if self.config.selection_mode == "flat":
            return torch.argmax(next_token_scores, dim=-1)

        cfg = self.config.token_layout
        out: list[int] = []

        for i, sid in enumerate(self._active_stream_ids):
            st = self._streams[sid]
            row = next_token_scores[i]

            normal = row[cfg.normal_byte_low : cfg.normal_byte_high + 1]
            fused = row[
                cfg.normal_byte_low + cfg.boundary_offset :
                cfg.normal_byte_high + cfg.boundary_offset + 1
            ]

            content_scores = torch.logaddexp(normal, fused)
            content_idx = int(torch.argmax(content_scores).item())

            phase_delta = float((fused[content_idx] - normal[content_idx]).item())

            h = float(self.config.phase_hysteresis)
            base_threshold = float(self.config.phase_threshold)

            if st.previous_boundary is None:
                threshold = base_threshold
            elif st.previous_boundary:
                threshold = base_threshold - h
            else:
                threshold = base_threshold + h

            boundary = phase_delta >= threshold

            token_id = cfg.offset + content_idx
            if boundary:
                token_id += cfg.boundary_offset
            out.append(token_id)

        return torch.tensor(out, dtype=torch.long, device=next_token_scores.device)

    def _push_domain_byte(
        self,
        buffer: list[int],
        cell_id: int,
        byte: int,
        packets: list[tuple[int, bytes]],
    ) -> None:
        buffer.append(int(byte) & 0xFF)
        if len(buffer) == self.config.word_size:
            packets.append((cell_id, pack_word4(*buffer)))
            buffer.clear()

    def _serialize_network_byte(self, metrics: PairedContentMetrics) -> int:
        return build_metric_byte(
            metrics.dominant_q6,
            bit0=(metrics.gauge_abs >= self.config.network_gauge_threshold),
            bit7=(metrics.support_ratio >= self.config.network_support_target),
        )

    def _serialize_database_byte(self, metrics: PairedContentMetrics) -> int:
        return build_metric_byte(
            metrics.q_support_mask,
            bit0=(metrics.peak_ratio >= self.config.database_drought_peak),
            bit7=(metrics.support_ratio >= self.config.database_tsunami_support),
        )

    def token_hook(
        self,
        next_tokens: torch.Tensor,
        boundary_offset: int,
    ) -> None:
        if next_tokens.ndim != 1:
            next_tokens = next_tokens.reshape(-1)
        if next_tokens.shape[0] != len(self._active_stream_ids):
            raise RuntimeError(
                "Decode bridge session batch_size does not match selected token batch"
            )

        packets: list[tuple[int, bytes]] = []
        cfg = self.config.token_layout

        for i, sid in enumerate(self._active_stream_ids):
            st = self._streams[sid]
            token_id = int(next_tokens[i].item())

            _, boundary = strip_boundary_phase(token_id, layout=cfg)
            canonical_byte = canonical_byte_from_token_id(token_id, layout=cfg)
            metrics = st.pending_metrics

            if canonical_byte is not None:
                st.current_patch_bytes.append(canonical_byte)
            if boundary:
                patch_idx = len(st.patch_records)
                payload = bytes(st.current_patch_bytes)
                st.patch_records.append(
                    PatchRecord(
                        patch_index=patch_idx,
                        bytes_payload=payload,
                        length=len(payload),
                        ended_by_boundary=True,
                    )
                )
                st.boundary_emit_count += 1
                st.current_patch_bytes.clear()
            if st.previous_boundary is not None and st.previous_boundary != boundary:
                st.gauge_flip_count += 1
            st.previous_boundary = boundary

            if metrics is not None:
                st.step_records.append(
                    PairedStepRecord(
                        step=st.step,
                        support_ratio=metrics.support_ratio,
                        peak_ratio=metrics.peak_ratio,
                        gauge_abs=metrics.gauge_abs,
                        dominant_q6=metrics.dominant_q6,
                        support_count=metrics.support_count,
                        raw_support_ratio=metrics.raw_support_ratio,
                        raw_support_count=metrics.raw_support_count,
                        selected_token_id=token_id,
                        selected_canonical_byte=canonical_byte,
                        selected_boundary=boundary,
                    )
                )

            if canonical_byte is None or metrics is None:
                st.step += 1
                continue

            application_byte = (
                shadow_partner_byte(canonical_byte)
                if boundary
                else canonical_byte
            )
            network_byte = self._serialize_network_byte(metrics)
            database_byte = self._serialize_database_byte(metrics)

            self._push_domain_byte(
                st.application_buffer,
                st.application_cell,
                application_byte,
                packets,
            )
            self._push_domain_byte(
                st.network_buffer,
                st.network_cell,
                network_byte,
                packets,
            )
            self._push_domain_byte(
                st.database_buffer,
                st.database_cell,
                database_byte,
                packets,
            )

            st.step += 1

        if packets:
            self.graph.ingest(packets)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _cell_order_parameters(self, cell_id: int) -> dict[str, Any]:
        chi_hist = self.graph._chi_hist64[cell_id].astype(np.float64, copy=False)
        shell_hist = self.graph._shell_hist7[cell_id].astype(np.float64, copy=False)
        total = float(chi_hist.sum())

        if total <= 0.0:
            return {
                "samples": 0,
                "rho": 0.0,
                "m": 0.0,
                "eta": 0.0,
                "effective_support_empirical": 0.0,
                "effective_support_theoretical": 0.0,
                "chi_support_ratio": 0.0,
                "chi_peak_ratio": 0.0,
                "shell_spectral": tuple(0.0 for _ in range(7)),
            }

        p = chi_hist / total
        p_nz = p[p > 0]
        effective_support_empirical = float(1.0 / np.square(p_nz).sum())
        chi_support_ratio = float(np.count_nonzero(chi_hist) / 64.0)
        chi_peak_ratio = float(p.max())

        mean_shell = float(np.dot(np.arange(7, dtype=np.float64), shell_hist) / total)
        rho = mean_shell / 6.0
        m = (2.0 * rho) - 1.0
        eta = -m
        effective_support_theoretical = float(4096.0 / ((1.0 + (eta * eta)) ** 6))

        shell_spectral = tuple(float(x) for x in self.graph.shell_spectral(cell_id))

        return {
            "samples": int(total),
            "rho": rho,
            "m": m,
            "eta": eta,
            "effective_support_empirical": effective_support_empirical,
            "effective_support_theoretical": effective_support_theoretical,
            "chi_support_ratio": chi_support_ratio,
            "chi_peak_ratio": chi_peak_ratio,
            "shell_spectral": shell_spectral,
        }

    def _stream_from_key(self, stream: str | int) -> _StreamState:
        if isinstance(stream, int):
            if stream < 0 or stream >= len(self._active_stream_ids):
                raise KeyError(f"Active stream index out of range: {stream}")
            return self._streams[self._active_stream_ids[stream]]
        if stream not in self._streams:
            raise KeyError(f"Unknown stream_id: {stream!r}")
        return self._streams[stream]

    def emit_stream_report(self, stream: str | int) -> BolmoDecodeReport:
        st = self._stream_from_key(stream)
        records = self.graph.emit_slcp(
            [st.network_cell, st.database_cell, st.application_cell]
        )
        lengths = [pr.length for pr in st.patch_records]
        total_steps = max(st.step, 1)
        support_ratios = [r.support_ratio for r in st.step_records]
        raw_support_ratios = [r.raw_support_ratio for r in st.step_records]
        support_counts = [r.support_count for r in st.step_records]
        raw_support_counts = [r.raw_support_count for r in st.step_records]
        gauge_abs_vals = [r.gauge_abs for r in st.step_records]
        patch_geometry = {
            "patch_count": len(st.patch_records),
            "attn_proxy": len(st.patch_records) * len(st.patch_records),
            "kv_proxy": len(st.patch_records),
            "mean_bytes_per_patch": (
                sum(lengths) / len(lengths) if lengths else 0.0
            ),
            "max_bytes_per_patch": max(lengths) if lengths else 0,
            "patch_length_hist": (
                list(lengths) if lengths else []
            ),
            "boundary_emit_count": st.boundary_emit_count,
            "gauge_flip_rate": st.gauge_flip_count / total_steps,
            "support_ratio_mean": (
                sum(support_ratios) / len(support_ratios) if support_ratios else 0.0
            ),
            "raw_support_ratio_mean": (
                sum(raw_support_ratios) / len(raw_support_ratios) if raw_support_ratios else 0.0
            ),
            "support_count_mean": (
                sum(support_counts) / len(support_counts) if support_counts else 0.0
            ),
            "raw_support_count_mean": (
                sum(raw_support_counts) / len(raw_support_counts) if raw_support_counts else 0.0
            ),
            "phase_redundancy_mean": (
                (sum(raw_support_counts) / len(raw_support_counts)) - (sum(support_counts) / len(support_counts))
                if raw_support_counts and support_counts
                else 0.0
            ),
            "gauge_abs_mean": (
                sum(gauge_abs_vals) / len(gauge_abs_vals) if gauge_abs_vals else 0.0
            ),
        }
        return BolmoDecodeReport(
            stream_id=st.stream_id,
            network={
                "record": records[0],
                "order": self._cell_order_parameters(st.network_cell),
            },
            database={
                "record": records[1],
                "order": self._cell_order_parameters(st.database_cell),
            },
            application={
                "record": records[2],
                "order": self._cell_order_parameters(st.application_cell),
            },
            patch_geometry=patch_geometry,
        )

    def emit_all_stream_reports(self) -> dict[str, BolmoDecodeReport]:
        out: dict[str, BolmoDecodeReport] = {}
        for sid in self._active_stream_ids:
            out[sid] = self.emit_stream_report(sid)
        return out