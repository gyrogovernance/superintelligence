"""Kernel-backed constants for the gyroscopic tools layer.

NOTE: this module is the tooling view of ``src.constants`` and ``constants.h``.
Do not invent new values here; only mirror from ``_k`` or C macros.
"""

from __future__ import annotations

import math
from src import constants as _k
# Re-exported kernel constants.
LAYER_MASK_12: int = _k.LAYER_MASK_12
MASK_STATE24: int = _k.MASK_STATE24
L0_MASK: int = _k.L0_MASK
LI_MASK: int = _k.LI_MASK
FG_MASK: int = _k.FG_MASK
BG_MASK: int = _k.BG_MASK
GENE_MIC_S: int = _k.GENE_MIC_S
GENE_MAC_A12: int = _k.GENE_MAC_A12
GENE_MAC_B12: int = _k.GENE_MAC_B12
GENE_MAC_REST: int = _k.GENE_MAC_REST
Q0: int = _k.Q0
Q1: int = _k.Q1
GATE_S_BYTES: tuple[int, int] = _k.GATE_S_BYTES
GATE_C_BYTES: tuple[int, int] = _k.GATE_C_BYTES
GATE_F_BYTES: tuple[int, ...] = GATE_S_BYTES + GATE_C_BYTES
HORIZON_GATE_BYTES: tuple[int, ...] = _k.HORIZON_GATE_BYTES
PAIR_MASKS_12: tuple[int, ...] = _k.PAIR_MASKS_12
GATE_NAMES: tuple[str, ...] = _k.GATE_NAMES
CHIRALITY_QUBITS_6: int = _k.CHIRALITY_QUBITS_6
CHIRALITY_MASK_6: int = _k.CHIRALITY_MASK_6
EPSILON_6: int = _k.EPSILON_6
OMEGA_SIZE: int = _k.OMEGA_SIZE
HORIZON_SIZE: int = _k.HORIZON_SIZE
BOUNDARY_SIZE: int = _k.BOUNDARY_SIZE
BULK_SIZE: int = _k.BULK_SIZE
DEPTH_CLOSURE: int = _k.DEPTH_CLOSURE
MASK_CODE_SIZE: int = _k.MASK_CODE_SIZE
SHADOW_PARTNER_MASK: int = _k.SHADOW_PARTNER_MASK
COMPLEMENT_MASK_12: int = _k.COMPLEMENT_MASK_12
SHADOW_STATES: int = _k.SHADOW_STATES
Q_G: float = _k.Q_G
DELTA_BU: float = _k.DELTA_BU
M_A: float = _k.M_A
RHO: float = _k.RHO
APERTURE_GAP: float = _k.APERTURE_GAP
APERTURE_GAP_Q256: int = _k.APERTURE_GAP_Q256

# Kernel convenience aliases.
pack_state = _k.pack_state
unpack_state = _k.unpack_state
byte_to_intron = _k.byte_to_intron
intron_family = _k.intron_family
intron_micro_ref = _k.intron_micro_ref
byte_family = _k.byte_family
byte_micro_ref = _k.byte_micro_ref
l0_parity = _k.l0_parity
li_parity = _k.li_parity
fg_parity = _k.fg_parity
bg_parity = _k.bg_parity
intron_cgm_parities = _k.intron_cgm_parities
byte_cgm_parities = _k.byte_cgm_parities
micro_ref_to_mask12 = _k.micro_ref_to_mask12
expand_intron_to_mask12 = _k.expand_intron_to_mask12
step_state_by_byte = _k.step_state_by_byte
inverse_step_by_byte = _k.inverse_step_by_byte
single_step_trace = _k.single_step_trace
popcount = _k.popcount
archetype_distance = _k.archetype_distance
horizon_distance = _k.horizon_distance
is_on_horizon = _k.is_on_horizon
is_on_equality_horizon = _k.is_on_equality_horizon
ab_distance = _k.ab_distance
complementarity_invariant = _k.complementarity_invariant
apply_gate_S = _k.apply_gate_S
apply_gate_C = _k.apply_gate_C
apply_gate_F = _k.apply_gate_F
apply_gate = _k.apply_gate
component_density = _k.component_density
vertex_charge_from_mask = _k.vertex_charge_from_mask
dot12 = _k.dot12

# Shared derived constants.
LAYER_BITS: int = _k.LAYER_BITS
FAMILY_MASK: int = _k.FAMILY_MASK
UINT8_MASK: int = _k.UINT8_MASK
UINT16_MASK: int = _k.UINT16_MASK
UINT32_MASK: int = _k.UINT32_MASK
# Keep the C mask value as-is so 64-bit platforms do not truncate UINT64_MASK.
UINT64_MASK: int = _k.UINT64_MASK
SHELL_MIDPOINT: int = _k.SHELL_MIDPOINT
SHELL_MAX_POPULATION: int = _k.SHELL_MAX_POPULATION
COMPLEMENTARITY_SUM: int = _k.COMPLEMENTARITY_SUM
SHELL_MAX: int = _k.SHELL_MAX
SHELL_COUNT: int = _k.SHELL_COUNT
GAUGE_COUNT: int = _k.GAUGE_COUNT
L0_BIT_0: int = _k.L0_BIT_0
L0_BIT_7: int = _k.L0_BIT_7
BYTE_COUNT: int = _k.BYTE_COUNT

# Process environment variables read by the gyroscopic backend or bench harness.
GYRO_ENV_VAR_NAMES: tuple[str, ...] = (
    "GGML_GYROSCOPIC",
    "GGML_GYROSCOPIC_STRICT",
    "GYROSCOPIC_TOTAL_LAYERS",
    "GYROSCOPIC_KV_CHI",
    "GYROSCOPIC_KV_CHI_INDEX",
    "GYROSCOPIC_GRAVITY_ATTN",
    "GYROSCOPIC_KV_CHI_STATS",
)

_C_HEADER_MAP: dict[str, int | float] = {
    "CHIRALITY_MASK_6": CHIRALITY_MASK_6,
    "LAYER_MASK_12": LAYER_MASK_12,
    "EPSILON_6": EPSILON_6,
    "COMPLEMENT_MASK_12": COMPLEMENT_MASK_12,
    "SHADOW_PARTNER_MASK": SHADOW_PARTNER_MASK,
    "Q_G": Q_G,
    "DEPTH_CLOSURE": DEPTH_CLOSURE,
    "MASK_CODE_SIZE": MASK_CODE_SIZE,
    "SHADOW_STATES": SHADOW_STATES,
    "L0_BIT_0": L0_BIT_0,
    "L0_BIT_7": L0_BIT_7,
    "BYTE_COUNT": BYTE_COUNT,
    "OMEGA_SIZE": OMEGA_SIZE,
    "HORIZON_SIZE": HORIZON_SIZE,
    "BOUNDARY_SIZE": BOUNDARY_SIZE,
    "BULK_SIZE": BULK_SIZE,
    "MASK_STATE24": MASK_STATE24,
    "SHELL_MAX": SHELL_MAX,
    "SHELL_MIDPOINT": SHELL_MIDPOINT,
    "SHELL_MAX_POPULATION": SHELL_MAX_POPULATION,
    "COMPLEMENTARITY_SUM": COMPLEMENTARITY_SUM,
    "UINT8_MASK": UINT8_MASK,
    "UINT16_MASK": UINT16_MASK,
    "UINT32_MASK": UINT32_MASK,
    "UINT64_MASK": UINT64_MASK,
    "GENE_MIC_S": GENE_MIC_S,
    "GENE_MAC_REST": GENE_MAC_REST,
    "GENE_MAC_A12": GENE_MAC_A12,
    "GENE_MAC_B12": GENE_MAC_B12,
    "CHIRALITY_QUBITS_6": CHIRALITY_QUBITS_6,
    "FAMILY_MASK": FAMILY_MASK,
    "LAYER_BITS": LAYER_BITS,
    "SHELL_COUNT": SHELL_COUNT,
    "GAUGE_COUNT": GAUGE_COUNT,
    "GYRO_M_PI": math.pi,
    "M_A": M_A,
    "DELTA_BU": DELTA_BU,
    "RHO": RHO,
    "APERTURE_GAP": APERTURE_GAP,
    "APERTURE_GAP_Q256": APERTURE_GAP_Q256,
}


def verify_c_header_sync(expected: dict[str, int | float] | None = None) -> list[str]:
    """Compare Python values against expected C-header values.

    Pass a dict of {macro_name: value} parsed from constants.h, or call with
    None to validate against _C_HEADER_MAP.

    Returns a list of mismatch descriptions, empty when matching.
    """

    reference = expected if expected is not None else _C_HEADER_MAP
    mismatches: list[str] = []
    for name, py_val in _C_HEADER_MAP.items():
        c_val = reference.get(name)
        if c_val is None:
            mismatches.append(f"{name}: missing from C header")
        elif py_val != c_val:
            mismatches.append(f"{name}: Python={py_val!r}, C={c_val!r}")
    for name in reference:
        if name not in _C_HEADER_MAP:
            mismatches.append(f"{name}: in C header but not in Python map")
    return mismatches


def assert_c_header_sync(expected: dict[str, int | float] | None = None) -> None:
    """Raise if Python constants diverge from parsed ``constants.h`` values.

    Call from CI or a tiny C program that dumps macro values for comparison.
    """
    mismatches = verify_c_header_sync(expected)
    if mismatches:
        raise AssertionError("constants.h mismatch:\n" + "\n".join(mismatches))