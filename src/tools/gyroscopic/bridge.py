from __future__ import annotations

import ctypes as ct
import hashlib

from src.constants import GENE_MAC_REST
from src.sdk import moment_from_ledger, state_charts

from .ops import (
    GyroGraphMoment,
    GyroGraphSLCP,
    GyroLabeOperatorReport,
    gyrograph_pack_moment,
    gyrograph_word_signature_from_bytes,
)


def verify_native_against_reference(
    cell_id: int,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    ledger_bytes: bytes,
) -> dict:
    try:
        from .ops import gyrograph_moment_from_ledger_native
    except Exception:
        gyrograph_moment_from_ledger_native = None

    if gyrograph_moment_from_ledger_native is not None:
        native_moment = gyrograph_moment_from_ledger_native(ledger_bytes)
    else:
        native_moment = gyrograph_pack_moment(
            cell_id, omega12_io, step_io, last_byte_io
        )
    ref_moment = moment_from_ledger(ledger_bytes, start_state24=GENE_MAC_REST)

    return {
        "step_match": native_moment.step == ref_moment.step,
        "state_match": native_moment.state24 == ref_moment.state24,
        "last_byte_match": native_moment.last_byte == ref_moment.last_byte,
        "parity_match": (
            native_moment.parity_O12 == ref_moment.parity_commitment[0]
            and native_moment.parity_E12 == ref_moment.parity_commitment[1]
        ),
        "q_match": native_moment.q_transport6 == ref_moment.q_transport6,
        "native": native_moment,
        "reference": ref_moment,
    }


def slcp_to_dict(slcp: GyroGraphSLCP) -> dict:
    return {
        "cell_id": slcp.cell_id,
        "step": slcp.step,
        "omega12": slcp.omega12,
        "state24": slcp.state24,
        "last_byte": slcp.last_byte,
        "family": slcp.family,
        "micro_ref": slcp.micro_ref,
        "q6": slcp.q6,
        "chi6": slcp.chi6,
        "shell": slcp.shell,
        "horizon_distance": slcp.horizon_distance,
        "ab_distance": slcp.ab_distance,
        "omega_sig": slcp.omega_sig,
        "parity_O12": slcp.parity_O12,
        "parity_E12": slcp.parity_E12,
        "parity_bit": slcp.parity_bit,
        "resonance_key": slcp.resonance_key,
        "current_resonance": slcp.current_resonance,
        "spectral64": list(slcp.spectral64),
        "gauge_spectral": list(slcp.gauge_spectral),
        "shell_spectral": list(slcp.shell_spectral),
    }


def build_result(moment: GyroGraphMoment, ledger_bytes: bytes) -> dict:
    charts = state_charts(moment.state24)
    sig = gyrograph_word_signature_from_bytes(ledger_bytes)
    return {
        "moment": {
            "t": int(moment.step),
            "s": int(moment.state24),
            "b": int(moment.last_byte),
            "Sigma": {
                "state_hex": charts.state_hex,
                "a_hex": charts.a_hex,
                "b_hex": charts.b_hex,
            },
        },
        "state": int(moment.state24),
        "charts": {
            "carrier": (
                int(moment.state24 >> 12) & 0xFFF,
                int(moment.state24) & 0xFFF,
            ),
            "chirality6": charts.chirality6,
            "constitutional": {
                "rest_distance": int(charts.constitutional.rest_distance),
                "horizon_distance": int(charts.constitutional.horizon_distance),
                "ab_distance": int(charts.constitutional.ab_distance),
                "is_on_horizon": bool(charts.constitutional.on_complement_horizon),
                "is_on_equality_horizon": bool(
                    charts.constitutional.on_equality_horizon
                ),
                "a_density": float(charts.constitutional.a_density),
                "b_density": float(charts.constitutional.b_density),
                "complementarity_sum": int(
                    charts.constitutional.complementarity_sum
                ),
            },
        },
        "provenance": {
            "archetype": 0xAA,
            "rest_state": GENE_MAC_REST,
            "ledger": ledger_bytes,
            "step_count": int(moment.step),
            "kernel_signature": (
                int(moment.step),
                int(moment.state24),
                int(moment.last_byte),
            ),
            "word_signature": {
                "parity": int(sig.parity),
                "tau_a12": int(sig.tau_a12),
                "tau_b12": int(sig.tau_b12),
            },
            "parity_commitment": (
                int(moment.parity_O12),
                int(moment.parity_E12),
                int(moment.parity_bit),
            ),
            "q_transport6": int(moment.q_transport6),
            "ledger_hash": hashlib.sha256(ledger_bytes).digest(),
        },
    }


def get_interoperability_outputs(
    slcp: GyroGraphSLCP,
    family_counts_256: list[int] | None = None,
    byte_ensemble_256: list[int] | None = None,
    operator_report: GyroLabeOperatorReport | None = None,
) -> dict:
    import numpy as np

    from .ops import gyrolabe_anisotropy_extract

    spec = np.asarray(list(slcp.spectral64), dtype=np.float64)
    nrm = float(np.linalg.norm(spec)) if spec.size else 0.0
    e = spec / 64.0 if spec.size else spec
    denom = float(np.sum(e * e)) if spec.size else 1.0
    eff_sup = 64.0 / denom if denom > 1e-12 else 0.0

    peak_mass = (
        float(spec[0] * spec[0]) / float(np.dot(spec, spec))
        if spec.size and float(np.dot(spec, spec)) > 1e-30
        else 0.0
    )
    spectral_damping = max(0.0, min(1.0, 1.0 - peak_mass))

    gs = [float(slcp.gauge_spectral[i]) for i in range(4)]
    gauge_anisotropy = [gs[0] - gs[2], gs[1] - gs[3]]

    if byte_ensemble_256 is not None and len(byte_ensemble_256) == 256:
        chi_anisotropy = gyrolabe_anisotropy_extract(byte_ensemble_256)
    elif family_counts_256 is not None and len(family_counts_256) == 256:
        chi_anisotropy = gyrolabe_anisotropy_extract(family_counts_256)
    else:
        chi_anisotropy = [float(spec[i]) / (nrm + 1e-12) for i in range(6)]

    if operator_report is not None:
        block_scr = float(operator_report.scr)
        block_defect_norm = float(operator_report.defect_norm)
        operator_class_id = int(operator_report.op_class)
        class_names = (
            "generic",
            "shell-radial",
            "shell-gauge",
            "chi-invariant",
            "chi-gauge",
        )
        block_class = (
            class_names[operator_class_id]
            if 0 <= operator_class_id < len(class_names)
            else "generic"
        )
    else:
        block_scr = nrm / 64.0 if spec.size else 0.0
        block_defect_norm = 0.0
        operator_class_id = -1
        block_class = "chi-invariant" if (spec.size and spec[0] > 0.9) else "generic"

    return {
        "block_class": block_class,
        "block_scr": block_scr,
        "block_defect_norm": block_defect_norm,
        "native_route": "gyrolabe_native:slcp+wht64+k4+k7",
        "kv_priority": 1.0 - float(slcp.current_resonance) / 64.0,
        "batch_group_id": int(slcp.chi6),
        "gauge_anisotropy": gauge_anisotropy,
        "spectral_damping": spectral_damping,
        "chi_anisotropy": chi_anisotropy,
        "effective_support": eff_sup,
        "shell_spectral": [float(slcp.shell_spectral[i]) for i in range(7)],
        "gauge_spectral": gs,
        "operator_class_id": operator_class_id,
    }
