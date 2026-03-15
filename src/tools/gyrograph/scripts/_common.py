from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from src.tools.gyrograph.bridges.applications import (
    ApplicationDecision,
    ApplicationsBridge,
)


def slcp_brief(rec) -> str:
    return (
        f"cell={rec.cell_id} step={rec.step} "
        f"omega12=0x{rec.omega12:03x} state24=0x{rec.state24:06x} "
        f"last=0x{rec.last_byte:02x} chi6=0x{rec.chi6:02x} shell={rec.shell} "
        f"q6=0x{rec.q6:02x} sig=0x{rec.omega_sig:04x} "
        f"res_key={rec.resonance_key} res_pop={rec.current_resonance}"
    )


def decision_brief(d: ApplicationDecision) -> str:
    return (
        f"entity={d.entity_id!r} role={d.role!r} cell={d.cell_id} "
        f"hot_loop_score={d.hot_loop_score:.4f} "
        f"contention_score={d.contention_score:.4f} "
        f"action={d.suggested_action} "
        f"res_key={d.resonance_key} res_pop={d.current_resonance} "
        f"shell={d.shell} chi6=0x{d.chi6:02x} "
        f"spec_norm={d.spectral_norm:.4f} "
        f"spec_peak_ratio={d.spectral_peak_ratio:.4f} "
        f"chi_support_ratio={d.chi_support_ratio:.4f} "
        f"chi_peak_ratio={d.chi_peak_ratio:.4f} "
        f"shell_entropy={d.shell_entropy:.4f}"
    )


def print_runtime_capabilities(bridge: ApplicationsBridge) -> None:
    caps = bridge.runtime_capabilities()
    print("[runtime capabilities]")
    for k in sorted(caps):
        print(f"  {k}: {caps[k]}")


def top_chi_buckets(bridge: ApplicationsBridge, entity_id: str, role: str = "main", top_k: int = 8):
    cid = bridge.entity_cell(entity_id, role)
    hist = bridge.graph._chi_hist64[cid]
    items = [(i, int(v)) for i, v in enumerate(hist) if int(v) > 0]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:top_k]


def shell_histogram(bridge: ApplicationsBridge, entity_id: str, role: str = "main") -> list[int]:
    cid = bridge.entity_cell(entity_id, role)
    return [int(x) for x in bridge.graph._shell_hist7[cid].tolist()]


def spectral_head(x: np.ndarray, n: int = 12) -> list[float]:
    return [float(v) for v in np.round(x[:n], 4).tolist()]


def spectral_tail(x: np.ndarray, n: int = 12) -> list[float]:
    if n <= 0:
        return []
    return [float(v) for v in np.round(x[-n:], 4).tolist()]


def top_resonance_buckets(bridge: ApplicationsBridge, top_k: int = 10) -> list[tuple[int, int]]:
    buckets = bridge.graph._resonance_buckets
    items = [(i, int(v)) for i, v in enumerate(buckets) if int(v) > 0]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:top_k]


def print_entity_probe(
    bridge: ApplicationsBridge,
    entity_id: str,
    role: str = "main",
    *,
    print_shell_spectral: bool = True,
) -> None:
    rec = bridge.emit_entity_slcp(entity_id, role)
    dec = bridge.profile_entity(entity_id, role)

    print("[SLCP]")
    print(" ", slcp_brief(rec))

    print("[decision]")
    print(" ", decision_brief(dec))

    print("[spectral]")
    print("  head:", spectral_head(rec.spectral64, 12))
    print("  tail:", spectral_tail(rec.spectral64, 12))
    print("  norm:", float(np.linalg.norm(rec.spectral64)))

    print("[histories]")
    print("  top chi buckets:", top_chi_buckets(bridge, entity_id, role))
    print("  shell histogram:", shell_histogram(bridge, entity_id, role))

    if print_shell_spectral:
        shell_spec = bridge.graph.shell_spectral(bridge.entity_cell(entity_id, role))
        print("  shell spectral:", shell_spec)

    print("[views]")
    charts = rec.charts()
    print("  state_hex:", charts.state_hex)
    print("  optical_coordinates:", rec.optical_coordinates())
    print("  stabilizer_type:", rec.stabilizer_type())


def decision_histogram(decisions: Iterable[ApplicationDecision]) -> dict[str, int]:
    ctr = Counter(d.suggested_action for d in decisions)
    return dict(sorted(ctr.items(), key=lambda kv: kv[0]))