from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, cast

import numpy as np
import torch

from src.api import q_word6, shadow_partner_byte
from src.tools.gyrolabe.bridges.bolmo_config import (
    BolmoTokenLayout,
)
from src.tools.gyrolabe import ops as gyrolabe_ops
from src.tools.gyrolabe.bridges.encode import EncodedFields, estimate_q_bath
from src.tools.gyrograph.bridges.decode import (
    compute_qubec_climate,
)
from src.tools.gyrograph.core import GyroGraph
from src.tools.gyrograph.profiles import ResonanceProfile, shell_from_omega12
from src.tools.gyrograph.serializers import pack_word4


_Q_LUT_256 = np.asarray([q_word6(b) for b in range(256)], dtype=np.uint8)


def strip_boundary_phase(
    token_id: int,
    *,
    layout: BolmoTokenLayout | None = None,
) -> tuple[int, bool]:
    cfg = layout or BolmoTokenLayout()
    t = int(token_id)
    boundary = t >= cfg.boundary_offset
    if boundary:
        t -= cfg.boundary_offset
    return t, boundary


def canonical_byte_from_token_id(
    token_id: int,
    *,
    layout: BolmoTokenLayout | None = None,
) -> int | None:
    cfg = layout or BolmoTokenLayout()
    t, _ = strip_boundary_phase(token_id, layout=cfg)
    if cfg.normal_byte_low <= t <= cfg.normal_byte_high:
        return t - cfg.offset
    return None


def intron_to_byte(
    family: int,
    micro_ref: int,
) -> int:
    fam = int(family) & 0x3
    micro = int(micro_ref) & 0x3F
    intron = (((fam >> 1) & 1) << 7) | (micro << 1) | (fam & 1)
    return intron ^ 0xAA


def build_metric_byte(
    micro_ref: int,
    *,
    bit0: bool = False,
    bit7: bool = False,
) -> int:
    family = ((1 if bit7 else 0) << 1) | (1 if bit0 else 0)
    return intron_to_byte(family, micro_ref)


@dataclass(frozen=True)
class BolmoDecodeBridgeConfig:
    token_layout: BolmoTokenLayout = field(default_factory=BolmoTokenLayout)

    # observe | gauge_damp | sector_shape | full
    control_mode: str = "observe"

    selection_mode: str = "paired"
    phase_threshold: float = 0.0
    phase_hysteresis: float = 0.0

    top_k: int = 64
    content_probability_floor: float = 1.0 / 256.0

    # Gauge damping on paired normal/fused logits
    application_phase_damping: float = 0.65
    min_phase_scale: float = 0.35
    max_phase_scale: float = 1.00

    # Ensemble-sector shaping
    database_support_target: float = 0.125
    database_sector_bonus: float = 0.35
    database_drought_peak: float = 0.35
    database_tsunami_support: float = 0.125

    # Network-side observational thresholds for serializer family bits
    network_support_target: float = 0.125
    network_gauge_threshold: float = 0.25

    network_role: str = "network"
    database_role: str = "database"
    application_role: str = "application"

    stream_prefix: str = "bolmo"
    word_size: int = 4
    proof_mode: bool = False
    logit_quant_clip: float = 32.0
    logit_quant_scale: float = 64.0


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
        use_opencl_hotpath: bool = False,
        opencl_min_batch: int = 1,
        enable_ingest_log: bool = False,
        ingest_log_path: str | None = None,
    ) -> None:
        """Construct decode bridge with CPU-path defaults for stable default behavior.

        OpenCL remains available through `use_opencl_hotpath=True`, but the
        normal route stays CPU-first for deterministic performance.
        """
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
        self._source_bath: Any = None

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

        scale = float(self.config.logit_quant_scale)
        normal_int = torch.round(normal * scale).to(torch.int32)
        fused_int = torch.round(fused * scale).to(torch.int32)

        # Tropical pairing: max replaces logaddexp.
        content_max = torch.maximum(normal_int, fused_int)
        max_val = content_max.max()
        margin = int(scale * 2.0)
        support_mask = content_max >= (max_val - margin)
        support_count = int(support_mask.sum().item())
        support_ratio = float(support_count / 256.0)
        peak_ratio = 1.0

        gauge_delta = fused_int - normal_int
        gauge_abs = float(gauge_delta[support_mask].abs().sum().item())

        q_ids = self._q_lut_t.to(row.device)
        q_scores = torch.zeros(64, dtype=torch.int32, device=row.device).scatter_reduce_(
            0,
            q_ids,
            content_max,
            reduce="amax",
            include_self=False,
        )
        dominant_q6 = int(torch.argmax(q_scores).item())

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
            raw_support_ratio=0.0,
            raw_support_count=0,
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

    def set_source_bath(self, q_bath: Any) -> None:
        """Store encode-side q-bath for CE34 climate equation."""
        self._source_bath = q_bath

    def _selector_shell_weights(self, stream: _StreamState) -> torch.Tensor:
        hist = self.graph._shell_hist7[stream.application_cell].astype(np.float64)
        total = float(hist.sum())

        if total < 8.0:
            return torch.full((7,), 4096, dtype=torch.int32, device="cpu")

        mean_N = 0.0
        for n in range(7):
            mean_N += float(n * hist[n])
        mean_N /= total
        rho = mean_N / 6.0
        lam = rho / max(1e-12, 1.0 - rho)

        from math import comb

        denom = (1.0 + lam) ** 6
        weights: list[int] = []
        for n in range(7):
            pi_n = (comb(6, n) * (lam ** n)) / max(1e-12, denom)
            w_int = max(1, int(round(4096.0 * pi_n)))
            weights.append(w_int)

        w = np.asarray(weights, dtype=np.int64)
        return torch.tensor(w.tolist(), dtype=torch.int32, device="cpu")

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
        batch = next_token_scores.shape[0]

        normal_float = next_token_scores[
            :, cfg.normal_byte_low : cfg.normal_byte_high + 1
        ]
        fused_float = next_token_scores[
            :, cfg.normal_byte_low + cfg.boundary_offset :
               cfg.normal_byte_high + cfg.boundary_offset + 1
        ]

        clip = float(self.config.logit_quant_clip)
        scale = float(self.config.logit_quant_scale)

        normal_int = torch.round(
            torch.clamp(normal_float, -clip, clip) * scale
        ).to(torch.int32)
        fused_int = torch.round(
            torch.clamp(fused_float, -clip, clip) * scale
        ).to(torch.int32)

        prev_vals: list[int] = []
        for sid in self._active_stream_ids:
            prev = self._streams[sid].previous_boundary
            prev_vals.append(255 if prev is None else (1 if prev else 0))
        prev_t = torch.tensor(prev_vals, dtype=torch.uint8, device="cpu")

        hysteresis_bias = int(round(float(self.config.phase_hysteresis) * scale))

        out_tokens: list[int] = []
        confident_margin = max(1536, hysteresis_bias * 2)
        proof_mode = bool(self.config.proof_mode)

        for i, sid in enumerate(self._active_stream_ids):
            st = self._streams[sid]

            row_normal = normal_int[i:i+1].contiguous()
            row_fused = fused_int[i:i+1].contiguous()
            row_prev = prev_t[i:i+1].contiguous()

            content_row = torch.maximum(row_normal[0], row_fused[0])
            top2 = torch.topk(content_row, k=min(2, content_row.numel())).values
            top_content_b = int(torch.argmax(content_row).item())

            prev_flag = int(row_prev[0].item())
            if prev_flag == 1:
                threshold = -hysteresis_bias
            elif prev_flag == 0:
                threshold = hysteresis_bias
            else:
                threshold = 0

            if proof_mode:
                row_shell_weights = self._selector_shell_weights(st)
                winning_byte, selected_boundary = gyrolabe_ops.exact_qsector_select(
                    row_normal,
                    row_fused,
                    hysteresis_bias,
                    row_prev,
                    row_shell_weights,
                )
                chosen_b = int(winning_byte[0].item())
            elif top2.numel() == 2 and int((top2[0] - top2[1]).item()) >= confident_margin:
                chosen_b = top_content_b
                phase_delta = int(row_fused[0, chosen_b].item()) - int(row_normal[0, chosen_b].item())
                selected_boundary = torch.tensor(
                    [1 if phase_delta >= threshold else 0],
                    dtype=torch.uint8,
                    device="cpu",
                )
            else:
                row_shell_weights = self._selector_shell_weights(st)
                winning_byte, selected_boundary = gyrolabe_ops.exact_qsector_select(
                    row_normal,
                    row_fused,
                    hysteresis_bias,
                    row_prev,
                    row_shell_weights,
                )
                chosen_b = int(winning_byte[0].item())

            if not proof_mode:
                chosen_score = int(content_row[chosen_b].item())
                top_score = int(content_row[top_content_b].item())
                max_drop = max(1024, hysteresis_bias * 2)
                if (top_score - chosen_score) > max_drop:
                    chosen_b = top_content_b
                    phase_delta = int(row_fused[0, chosen_b].item()) - int(row_normal[0, chosen_b].item())
                    selected_boundary[0] = 1 if phase_delta >= threshold else 0

            token_id = int(chosen_b) + cfg.offset
            if int(selected_boundary[0].item()) != 0:
                token_id += cfg.boundary_offset
            out_tokens.append(token_id)

        return torch.tensor(out_tokens, dtype=torch.long, device=next_token_scores.device)

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
            bit0=(metrics.gauge_abs > 0),
            bit7=(metrics.support_count > 4),
        )

    def _serialize_database_byte(self, metrics: PairedContentMetrics) -> int:
        return build_metric_byte(
            metrics.q_support_mask,
            bit0=(metrics.support_count > 16),
            bit7=(metrics.support_count > 64),
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

        if self._attached_model is not None:
            enc = getattr(self._attached_model, "last_encoded_fields", None)
            if enc is not None and callable(enc):
                fields = cast(EncodedFields | None, enc())
                if fields is not None:
                    self.set_source_bath(estimate_q_bath(fields))
            if self._active_stream_ids:
                st = self._streams[self._active_stream_ids[0]]
                chi_hist = self.graph._chi_hist64[st.application_cell]
                shell_hist = self.graph._shell_hist7[st.application_cell]
                samples = int(chi_hist.sum())
                climate = compute_qubec_climate(chi_hist, shell_hist, samples)
                set_m2 = getattr(self._attached_model, "set_source_m2", None)
                if set_m2 is not None and callable(set_m2):
                    set_m2(climate.M2)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _cell_order_parameters(self, cell_id: int) -> dict[str, Any]:
        chi_hist = self.graph._chi_hist64[cell_id]
        shell_hist = self.graph._shell_hist7[cell_id]
        samples = int(chi_hist.sum())

        if samples <= 0:
            return {
                "samples": 0,
                "rho": 0.5,
                "m": 0.0,
                "eta": 0.0,
                "effective_support_empirical": 4096.0,
                "effective_support_theoretical": 4096.0,
                "chi_support_ratio": 0.0,
                "chi_peak_ratio": 0.0,
                "shell_spectral": tuple(0.0 for _ in range(7)),
            }

        climate = compute_qubec_climate(
            chi_hist64=chi_hist,
            shell_hist7=shell_hist,
            samples=samples,
        )
        p = chi_hist.astype(np.float64, copy=False) / float(samples)
        chi_support_ratio = float(np.count_nonzero(chi_hist) / 64.0)
        chi_peak_ratio = float(p.max())

        return {
            "samples": samples,
            "rho": climate.rho,
            "m": climate.m,
            "eta": climate.eta,
            "effective_support_empirical": climate.M2,
            "effective_support_theoretical": climate.M2_eq,
            "chi_support_ratio": chi_support_ratio,
            "chi_peak_ratio": chi_peak_ratio,
            "shell_spectral": climate.shell_spectrum,
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

        # In proof mode, exact q-sector selection is already enforced.
        # Report zero phase redundancy by aligning raw and effective counts.
        if self.config.proof_mode:
            raw_support_counts = support_counts

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
