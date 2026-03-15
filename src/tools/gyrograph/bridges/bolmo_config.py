from __future__ import annotations

from dataclasses import dataclass, field

from src.tools.gyrolabe.bridges.bolmo_config import (
    DEFAULT_BOLMO_MODEL_PATH,
    BolmoTokenLayout,
)


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