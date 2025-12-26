"""
Physics diagnostic experiment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from baby.kernel.gyro_core import GyroEngine


def load_engine_from_config() -> GyroEngine:
    config_path = project_root / "baby" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    engine = GyroEngine(
        atlas_paths=cfg["atlas"],
        store_paths=cfg.get("stores", {}),
        runtime=cfg.get("runtime", {}),
        version_info=cfg.get("version", {}),
    )
    return engine


def compute_information_quantum() -> float:
    # I_CGM = log2(N*) * S_min/m_p = log2(37) * (pi/2)
    N_star = 37
    S_min_over_mp = np.pi / 2.0
    return float(np.log2(N_star) * S_min_over_mp)


def compute_critical_info_density() -> float:
    # ρ_info = N* / Q_G = 37 / (4π)
    N_star = 37.0
    Q_G = 4.0 * np.pi
    return float(N_star / Q_G)


def compute_intelligence_quantum() -> float:
    # IQ_CGM = log2(788,986) / log2(256)
    return float(np.log2(788_986) / np.log2(256))


def summarize_theta(engine: GyroEngine) -> Tuple[float, float, float]:
    # min, mean, max theta
    th = engine.theta
    return float(np.min(th)), float(np.mean(th)), float(np.max(th))


def sample_transitions_theta_phase(engine: GyroEngine, num_states: int = 2048, intron: int = 0x55) -> Dict[str, float]:
    # Sample a subset of states, apply one intron via ep table, report delta theta and phase stats
    n_total = len(engine.keys)
    step = max(1, n_total // num_states)
    indices = np.arange(0, n_total, step, dtype=np.int64)
    if len(indices) > num_states:
        indices = indices[:num_states]

    theta = engine.theta
    keys = engine.keys

    deltas: List[float] = []
    phases_before: List[int] = []
    phases_after: List[int] = []

    for idx in indices:
        state = int(keys[idx])
        theta_before = float(theta[idx])
        phase_before, _ = engine.compute_state_phase(state)
        new_idx = engine.apply_intron_index(int(idx), intron)
        state_after = int(keys[new_idx])
        theta_after = float(theta[new_idx])
        phase_after, _ = engine.compute_state_phase(state_after)

        deltas.append(theta_after - theta_before)
        phases_before.append(phase_before)
        phases_after.append(phase_after)

    deltas_np = np.asarray(deltas, dtype=np.float64)
    pb = np.asarray(phases_before, dtype=np.int32)
    pa = np.asarray(phases_after, dtype=np.int32)

    # Circular distance on 8-bit ring for phase change
    diff = np.abs(pa - pb)
    phase_change = np.minimum(diff, 256 - diff)

    return {
        "samples": float(len(indices)),
        "theta_delta_mean": float(np.mean(deltas_np)),
        "theta_delta_std": float(np.std(deltas_np)),
        "theta_delta_min": float(np.min(deltas_np)),
        "theta_delta_max": float(np.max(deltas_np)),
        "phase_change_mean": float(np.mean(phase_change)),
        "phase_change_std": float(np.std(phase_change)),
        "phase_change_min": float(np.min(phase_change)),
        "phase_change_max": float(np.max(phase_change)),
    }


def test_information_geometry_bridge(engine: GyroEngine) -> Dict[str, float]:
    # Notes: θ_posterior = gyr[θ_evidence, θ_prior] maps operationally to applying an intron
    # We'll emulate: choose random states and a fixed intron, measure θ changes
    stats = sample_transitions_theta_phase(engine, num_states=2048, intron=0x55)
    return stats


def test_orbit_properties(engine: GyroEngine) -> Dict[str, float]:
    # Basic orbit stats: unique reps, mean orbit size, and coverage ratio
    reps = np.unique(engine.pheno)
    orbit_sizes = engine.orbit_sizes
    return {
        "num_representatives": float(len(reps)),
        "mean_orbit_size": float(np.mean(orbit_sizes)),
        "min_orbit_size": float(np.min(orbit_sizes)),
        "max_orbit_size": float(np.max(orbit_sizes)),
        "coverage_fraction": float(np.sum(orbit_sizes > 0) / len(orbit_sizes)),
    }


def main() -> None:
    engine = load_engine_from_config()

    # Constants from Notes_1.md
    I_cgm = compute_information_quantum()
    rho_info = compute_critical_info_density()
    IQ_cgm = compute_intelligence_quantum()

    th_min, th_mean, th_max = summarize_theta(engine)

    # Inference-as-geometry sampling
    ig_stats = test_information_geometry_bridge(engine)

    # Orbit properties
    orbit_stats = test_orbit_properties(engine)

    # Print concise report
    print("=== CGM ↔ GyroSI Diagnostic (Notes_1.md) ===")
    print(f"I_CGM (bits): {I_cgm:.3f}  [expected ~8.18]")
    print(f"ρ_info (bits/steradian): {rho_info:.3f}  [expected ~2.94]")
    print(f"IQ_CGM (layers): {IQ_cgm:.3f}  [expected ~2.45]")
    print()

    print("-- θ statistics --")
    print(f"θ_min={th_min:.6f}  θ_mean={th_mean:.6f}  θ_max={th_max:.6f}")
    print()

    print("-- Inference-as-gyration (single intron 0x55) --")
    print(
        "samples={samples:.0f}  Δθ_mean={theta_delta_mean:.6f}  Δθ_std={theta_delta_std:.6f}  "
        "Δθ_min={theta_delta_min:.6f}  Δθ_max={theta_delta_max:.6f}".format(**ig_stats)
    )
    print(
        "phase_change_mean={phase_change_mean:.3f}  phase_change_std={phase_change_std:.3f}  "
        "phase_change_min={phase_change_min:.0f}  phase_change_max={phase_change_max:.0f}".format(**ig_stats)
    )
    print()

    print("-- Orbit properties --")
    print(
        "reps={num_representatives:.0f}  mean_size={mean_orbit_size:.2f}  min_size={min_orbit_size:.0f}  "
        "max_size={max_orbit_size:.0f}  coverage={coverage_fraction:.3f}".format(**orbit_stats)
    )


if __name__ == "__main__":
    main()

