"""Dataset schemas, census builders, dense tables, and versioned manifests.

Generates and loads kernel-derived datasets under ``data/`` (gitignored;
regenerable via ``cli generate``). Includes the CGM Null Dataset
(``dataset_null``) and ``NullCorpus`` splits. Null construction starts at
archetype ``GENE_MIC_S = 0xAA`` and rest ``GENE_MAC_REST``, then builds
``byte_fiber``, ``canonical_cycles``, ``depth2_witnesses``,
``signature_words``, ``collisions``, and ``measures.json`` (see README
"CGM Null Dataset"). Values come from the hQVM kernel; this module reshapes
kernel calls into arrays and records schema plus invariant checks in a JSON
manifest. Signature id packing: ``kernel.signature_id`` /
``kernel.signature_from_id``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src import api, constants
from src.constants import (
    GENE_MAC_A12,
    GENE_MAC_B12,
    GENE_MAC_REST,
    GENE_MIC_S,
    step_state_by_byte,
)
from src.family import (
    byte_from_family_micro,
    fold_disagreement_d,
    intron_family_d,
    intron_from_byte,
    intron_micro_ref_d,
)

from . import kernel
from . import paths

# One folder per dataset under data/dataset_<name>/ (see paths.dataset_dir).
# The five kernel-derived tables use these one-word keys:
#   bytes, states, transitions, signatures, actions

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Fixed shapes
# ---------------------------------------------------------------------------

N_BYTES = 256
N_STATES = 4096
N_TRANSITIONS = N_STATES * N_BYTES  # 1,048,576
N_SIGNATURES = 8192  # |G| = (2^6 x 2^6) semidirect C2
N_SHELLS = 7

# ---------------------------------------------------------------------------
# Byte census (4.1): one row per byte
# ---------------------------------------------------------------------------

BYTE_CENSUS_FIELDS: dict[str, Any] = {
    "byte_u8": np.uint8,
    "intron_u8": np.uint8,
    "family_u2": np.uint8,  # 2-bit family in 0..3
    "family_bit0": np.uint8,
    "family_bit7": np.uint8,
    "micro_ref_u6": np.uint8,  # 6-bit payload in 0..63
    "mask12": np.uint16,
    "q6": np.uint8,
    "q_weight": np.uint8,
    "l0_parity": np.uint8,
    "li_parity": np.uint8,
    "fg_parity": np.uint8,
    "bg_parity": np.uint8,
    "fold_disagreement": np.uint8,  # 0..4 phase-pair disagreements
    "shadow_partner_byte": np.uint8,
    "shadow_pair_id": np.uint16,  # min(byte, partner); ids 0..127
    "eps_a6": np.uint8,
    "eps_b6": np.uint8,
}

# ---------------------------------------------------------------------------
# State census (4.2): one row per Omega state
# ---------------------------------------------------------------------------

STATE_CENSUS_FIELDS: dict[str, Any] = {
    "state_index": np.uint16,  # (u6 << 6) | v6
    "packed_omega12": np.uint16,
    "state24": np.uint32,
    "a12": np.uint16,
    "b12": np.uint16,
    "u6": np.uint8,
    "v6": np.uint8,
    "chirality6": np.uint8,
    "shell_chi": np.uint8,  # popcount(chirality6); 0 equality, 6 complement side
    "arch_shell": np.uint8,  # 6 - shell_chi
    "equality_horizon": np.uint8,  # bool
    "complement_horizon": np.uint8,  # bool
    "bulk": np.uint8,  # bool
    "optical_eq_num": np.uint8,  # Fraction(shell, 6) stored as exact ints
    "optical_eq_den": np.uint8,
    "optical_comp_num": np.uint8,
    "optical_comp_den": np.uint8,
    "optical_mu_num": np.int8,  # (2*shell - 6)/6; numerator can be negative
    "optical_mu_den": np.uint8,
    "stabilizer_type": np.uint8,  # 0 equality, 1 complement, 2 bulk
    "k4_stabilizer_mask": np.uint8,  # bit0 S, bit1 C, bit2 F
    "k4_orbit_id": np.uint16,
    "k4_orbit_size": np.uint8,
    "spin_a6": np.uint16,  # spin A6 packed (pair 10 -> 1, 01 -> 0 per 2 bits)
    "spin_b6": np.uint16,
}

# ---------------------------------------------------------------------------
# Dense transition tables (4.3)
# ---------------------------------------------------------------------------

TRANSITION_DTYPES: dict[str, Any] = {
    "next_state": np.uint16,  # [4096, 256] state_index of destination
    "inverse_state": np.uint16,  # [4096, 256] state_index of predecessor
}

# ---------------------------------------------------------------------------
# K4 action dataset (4.4): [4, 4096] permutation rows for id, S, C, F
# ---------------------------------------------------------------------------

K4_GATES = ("id", "S", "C", "F")
K4_ACTION_FIELDS: dict[str, Any] = {
    "action": np.uint16,  # [4, 4096] dest state_index per (gate, state)
    "fixed": np.uint8,  # [4, 4096] bool: gate fixes state
}

# ---------------------------------------------------------------------------
# Group signature dataset (4.5): 8192 rows
# ---------------------------------------------------------------------------

SIGNATURE_FIELDS: dict[str, Any] = {
    "sig_id": np.uint16,  # (parity << 12) | (tau_u6 << 6) | tau_v6
    "parity": np.uint8,
    "tau_u6": np.uint8,
    "tau_v6": np.uint8,
    "inverse_sig_id": np.uint16,
    "is_translation": np.uint8,  # parity 0
    "is_swap": np.uint8,  # parity 1 (any odd affine signature)
    "is_central_swap": np.uint8,  # parity 1, tau = 0 (bare coordinate swap)
    "action_on_rest": np.uint16,  # state_index of signature applied to rest
    "chi_increment_u": np.uint8,  # tau_u6 ^ tau_v6 (chirality shift)
    "chi_increment_v": np.uint8,  # identical to chi_increment_u (kernel law)
}


# ---------------------------------------------------------------------------
# Census builders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def byte_census_arrays() -> dict[str, np.ndarray]:
    """Build the 256-row byte census."""
    n = 256
    out: dict[str, np.ndarray] = {}
    out["byte_u8"] = np.arange(n, dtype=np.uint8)
    introns = np.array(api.INTRON_BY_BYTE, dtype=np.uint8)
    out["intron_u8"] = introns
    out["family_u2"] = np.array(api.FAMILY_BY_BYTE, dtype=np.uint8)
    out["family_bit0"] = (introns & 1).astype(np.uint8)
    out["family_bit7"] = ((introns >> 7) & 1).astype(np.uint8)
    out["micro_ref_u6"] = np.array(api.MICRO_REF_BY_BYTE, dtype=np.uint8)
    out["mask12"] = np.array(api.MASK12_BY_BYTE, dtype=np.uint16)
    out["q6"] = np.array([api.q_word6(b) for b in range(n)], dtype=np.uint8)
    out["q_weight"] = np.array(api.Q_WEIGHT_BY_BYTE, dtype=np.uint8)
    parities = [constants.byte_cgm_parities(b) for b in range(n)]
    out["l0_parity"] = np.array([p["L0"] for p in parities], dtype=np.uint8)
    out["li_parity"] = np.array([p["LI"] for p in parities], dtype=np.uint8)
    out["fg_parity"] = np.array([p["FG"] for p in parities], dtype=np.uint8)
    out["bg_parity"] = np.array([p["BG"] for p in parities], dtype=np.uint8)
    out["fold_disagreement"] = np.array(
        [fold_disagreement_d(b, 6) for b in range(n)], dtype=np.uint8
    )
    out["shadow_partner_byte"] = np.array(api.SHADOW_PARTNER_BY_BYTE, dtype=np.uint8)
    out["shadow_pair_id"] = np.array(
        [min(b, api.shadow_partner_byte(b)) for b in range(n)], dtype=np.uint16
    )
    out["eps_a6"] = np.array(api.EPS_A6_BY_BYTE, dtype=np.uint8)
    out["eps_b6"] = np.array(api.EPS_B6_BY_BYTE, dtype=np.uint8)
    return out


@lru_cache(maxsize=4)
def state_census_arrays() -> dict[str, np.ndarray]:
    """Build the 4096-row state census."""
    n = 4096
    out: dict[str, np.ndarray] = {
        name: np.zeros(n, dtype=dtype)
        for name, dtype in STATE_CENSUS_FIELDS.items()
    }

    orbit_ids: dict[frozenset[int], int] = {}
    for index in range(n):
        state24 = int(kernel.state24_from_index(index))
        omega = api.state24_to_omega12(state24)
        a12, b12 = constants.unpack_state(state24)
        chi = omega.chirality6
        shell_chi = chi.bit_count()
        arch_shell = 6 - shell_chi
        eq = omega.is_on_equality_horizon
        comp = omega.is_on_complement_horizon

        out["state_index"][index] = index
        out["packed_omega12"][index] = (omega.u6 << 6) | omega.v6
        out["state24"][index] = state24
        out["a12"][index] = a12
        out["b12"][index] = b12
        out["u6"][index] = omega.u6
        out["v6"][index] = omega.v6
        out["chirality6"][index] = chi
        out["shell_chi"][index] = shell_chi
        out["arch_shell"][index] = arch_shell
        out["equality_horizon"][index] = int(eq)
        out["complement_horizon"][index] = int(comp)
        out["bulk"][index] = int(not (eq or comp))
        out["optical_eq_num"][index] = shell_chi
        out["optical_eq_den"][index] = 6
        out["optical_comp_num"][index] = 6 - shell_chi
        out["optical_comp_den"][index] = 6
        out["optical_mu_num"][index] = 2 * shell_chi - 6
        out["optical_mu_den"][index] = 6
        stab = api.stabilizer_type_from_state24(state24)
        out["stabilizer_type"][index] = {"equality": 0, "complement": 1, "bulk": 2}[stab]
        stab_mask = 0
        if "S" in api.k4_stabilizer(state24):
            stab_mask |= 1
        if "C" in api.k4_stabilizer(state24):
            stab_mask |= 2
        if "F" in api.k4_stabilizer(state24):
            stab_mask |= 4
        out["k4_stabilizer_mask"][index] = stab_mask
        orbit = api.k4_orbit(state24)
        if orbit not in orbit_ids:
            orbit_ids[orbit] = len(orbit_ids)
        out["k4_orbit_id"][index] = orbit_ids[orbit]
        out["k4_orbit_size"][index] = len(orbit)

        spin_a, spin_b = api.state24_to_spin6_pair(state24)
        out["spin_a6"][index] = _pack_spins6(spin_a)
        out["spin_b6"][index] = _pack_spins6(spin_b)
    return out


def _pack_spins6(spins: tuple[int, ...]) -> int:
    """Pack +/-1 spins into 2 bits each (10=+1, 01=-1) for uint16 storage."""
    packed = 0
    for i, s in enumerate(spins):
        packed |= (0b10 if s == 1 else 0b01) << (2 * i)
    return packed


# ---------------------------------------------------------------------------
# Transition tables
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def transition_table() -> np.ndarray:
    """next_state[4096, 256] -> canonical state index of destination.

    Cached: the 4096 x 256 kernel build is ~1M calls, and many readouts,
    datasets, and replay routines reuse it. The table is immutable (pure
    kernel function of state + byte), so a single cached instance is correct.
    """
    table = np.empty((4096, 256), dtype=np.uint16)
    for index in range(4096):
        state24 = int(kernel.state24_from_index(index))
        omega = api.state24_to_omega12(state24)
        for byte in range(256):
            dest = api.step_omega12_by_byte(omega, byte)
            table[index, byte] = (dest.u6 << 6) | dest.v6
    table.setflags(write=False)
    return table


def inverse_transition_table() -> np.ndarray:
    """inverse_state[4096, 256] -> canonical index of the predecessor."""
    return _build_inverse_transition_table()


@lru_cache(maxsize=1)
def _build_inverse_transition_table() -> np.ndarray:
    table = np.empty((4096, 256), dtype=np.uint16)
    for index in range(4096):
        state24 = int(kernel.state24_from_index(index))
        for byte in range(256):
            table[index, byte] = kernel.state_index(
                constants.inverse_step_by_byte(state24, byte)
            )
    table.setflags(write=False)
    return table


# ---------------------------------------------------------------------------
# Signature dataset
# ---------------------------------------------------------------------------


def signature_dataset() -> dict[str, np.ndarray]:
    """8192-row full-G signature table (composition computed on demand)."""
    n = 8192
    out: dict[str, np.ndarray] = {
        name: np.zeros(n, dtype=dtype)
        for name, dtype in SIGNATURE_FIELDS.items()
    }
    rest = api.state24_to_omega12(constants.GENE_MAC_REST)
    for sig_id in range(n):
        parity, tau_u6, tau_v6 = kernel.sig_id_parts(sig_id)
        out["sig_id"][sig_id] = sig_id
        out["parity"][sig_id] = parity
        out["tau_u6"][sig_id] = tau_u6
        out["tau_v6"][sig_id] = tau_v6
        out["inverse_sig_id"][sig_id] = kernel.signature_inverse_id(sig_id)
        out["is_translation"][sig_id] = int(parity == 0)
        out["is_swap"][sig_id] = int(parity == 1)
        out["is_central_swap"][sig_id] = int(parity == 1 and tau_u6 == 0 and tau_v6 == 0)
        dest = api.apply_omega_signature(rest, api.OmegaSignature12(parity, tau_u6, tau_v6))
        out["action_on_rest"][sig_id] = (dest.u6 << 6) | dest.v6
        out["chi_increment_u"][sig_id] = tau_u6 ^ tau_v6
        out["chi_increment_v"][sig_id] = tau_u6 ^ tau_v6
    return out


# ---------------------------------------------------------------------------
# Dataset writer with manifest
# ---------------------------------------------------------------------------


@dataclass
class GeneratedDataset:
    name: str
    arrays: dict[str, np.ndarray]
    manifest: DatasetManifest

    def save(self, data_dir: Path | None = None) -> Path:
        out_dir = data_dir if data_dir is not None else paths.dataset_dir(self.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest.arrays = {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size_bytes": DatasetManifest.entry_bytes(
                    list(arr.shape), str(arr.dtype)
                ),
            }
            for name, arr in self.arrays.items()
        }
        self.manifest.row_count = int(self.arrays[next(iter(self.arrays))].shape[0])
        self.manifest.kernel_fingerprint = DatasetManifest.kernel_fingerprint_of(
            Path(constants.__file__).parent
        )
        for name, arr in self.arrays.items():
            np.save(out_dir / f"{name}.npy", arr)
        self.manifest.to_json(out_dir / "manifest.json")
        return out_dir


def run_invariant_checks(name: str, arrays: dict[str, np.ndarray]) -> dict[str, bool]:
    """Fast invariant checks embedded in the manifest of each dataset."""
    checks: dict[str, bool] = {}
    if name == "states":
        shell_chi = arrays["shell_chi"]
        checks["arch_shell_complement"] = bool(
            np.array_equal(arrays["arch_shell"], 6 - shell_chi)
        )
        checks["population_64C6w"] = bool(
            np.array_equal(
                np.bincount(shell_chi, minlength=7),
                np.array([64 * _comb(6, w) for w in range(7)]),
            )
        )
        checks["horizons_64_each"] = bool(
            int(arrays["equality_horizon"].sum()) == 64
            and int(arrays["complement_horizon"].sum()) == 64
        )
        checks["bulk_3968"] = bool(int(arrays["bulk"].sum()) == 3968)
    if name == "transitions":
        nxt = arrays["next_state"]
        inv = arrays["inverse_state"]
        idx = np.arange(4096, dtype=np.uint16)[:, None]
        byt = np.arange(256, dtype=np.uint16)[None, :]
        checks["inverse_roundtrip"] = bool(
            np.array_equal(inv[nxt[idx, byt], byt], idx)
        )
        checks["range_valid"] = bool(nxt.max() < 4096 and inv.max() < 4096)
    if name == "actions":
        action = arrays["action"]
        checks["id_identity"] = bool(np.array_equal(action[0], np.arange(4096)))
        checks["involutions"] = bool(
            np.array_equal(action[1][action[1]], np.arange(4096))
            and np.array_equal(action[2][action[2]], np.arange(4096))
            and np.array_equal(action[3][action[3]], np.arange(4096))
        )
    if name == "signatures":
        checks["count_8192"] = bool(arrays["sig_id"].shape[0] == 8192)
        inv = arrays["inverse_sig_id"]
        checks["involutions_consistent"] = bool(
            np.array_equal(inv[inv], np.arange(8192, dtype=inv.dtype))
        )
    return checks


def _comb(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


def generate_dataset(name: str, data_dir: Path | None = None) -> Path:
    """Generate one named dataset with its manifest.

    With ``data_dir`` given the arrays land in ``data_dir / name`` (used by the
    tests for isolation); otherwise they land in ``paths.dataset_dir(name)``
    (the repo layout ``data/dataset_<name>/``). The ``null`` corpus always
    lands in ``paths.dataset_dir("null")`` (or ``data_dir / "dataset_null"``
    when a custom home is given)."""
    if name == "null":
        target = (
            Path(data_dir) / "dataset_null"
            if data_dir is not None
            else paths.dataset_dir("null")
        )
        return generate_null_dataset(target)

    target = Path(data_dir) / name if data_dir is not None else paths.dataset_dir(name)
    if name == "bytes":
        arrays = byte_census_arrays()
    elif name == "states":
        arrays = state_census_arrays()
    elif name == "transitions":
        arrays = {"next_state": transition_table(), "inverse_state": inverse_transition_table()}
    elif name == "actions":
        action, fixed = kernel.k4_action_arrays()
        arrays = {"action": action, "fixed": fixed}
    elif name == "signatures":
        arrays = signature_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    dataset = GeneratedDataset(
        name=name,
        arrays=arrays,
        manifest=DatasetManifest(
            dataset_name=name,
            checks=run_invariant_checks(name, arrays),
        ),
    )
    return dataset.save(target)


def generate_all(data_dir: Path | None = None) -> list[Path]:
    return [generate_dataset(name, data_dir) for name in (
        "bytes", "states", "transitions", "actions", "signatures", "null"
    )]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_dataset(name: str, data_dir: Path | None = None) -> dict[str, np.ndarray]:
    out_dir = Path(data_dir) / name if data_dir is not None else paths.dataset_dir(name)
    if not out_dir.exists():
        generate_dataset(name, data_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failed = [k for k, v in manifest.get("checks", {}).items() if not v]
        if failed:
            raise ValueError(f"Dataset {name!r} failed invariant checks: {failed}")
        # A stale dataset (generated by a different kernel or environment)
        # would silently drift from the live kernel; regenerate on mismatch.
        fp = manifest.get("kernel_fingerprint")
        if fp and fp != DatasetManifest.kernel_fingerprint_of(
            Path(constants.__file__).parent
        ):
            out_dir = generate_dataset(name, data_dir)
    return {
        p.stem: np.load(p)
        for p in sorted(out_dir.glob("*.npy"))
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class DatasetManifest:
    """Versioned manifest emitted alongside every generated dataset."""

    dataset_name: str
    schema_version: str = SCHEMA_VERSION
    config: dict = field(default_factory=dict)
    kernel_fingerprint: str = ""
    row_count: int = 0
    arrays: dict = field(default_factory=dict)  # name -> {shape, dtype, size_bytes}
    checks: dict = field(default_factory=dict)  # invariant name -> bool
    seed: int = 0

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    @staticmethod
    def entry_bytes(shape: list[int], dtype: str) -> int:
        """Total byte size of an array for the manifest, using the actual
        numpy itemsize of the declared dtype."""
        return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)

    @staticmethod
    def kernel_fingerprint_of(src_dir: Path) -> str:
        """Stable hash of the kernel modules this package depends on."""
        digest = hashlib.sha256()
        for name in ("api.py", "constants.py", "family.py"):
            module_path = src_dir / name
            if module_path.exists():
                digest.update(module_path.read_bytes())
        digest.update((platform.python_version() + np.__version__).encode())
        return digest.hexdigest()[:16]

# ---------------------------------------------------------------------------
# Null corpus (dataset_null): permutation atlas + NullCorpus splits
# ---------------------------------------------------------------------------

D = 6
GENE_MAC_SWAPPED = (GENE_MAC_B12 << 12) | GENE_MAC_A12
FINITE_APERTURE = 5 / 256
DEPTH4_ALIGNMENT = 48 * FINITE_APERTURE

BYTE_FIBER_DTYPE = np.dtype(
    [
        ("byte", np.uint8),
        ("intron", np.uint8),
        ("family_phase", np.uint8),
        ("micro_ref", np.uint8),
        ("q6", np.uint8),
        ("q_weight", np.uint8),
        ("fwd4", np.uint8),
        ("rev4", np.uint8),
        ("fold_disagreement", np.uint8),
        ("is_flat", np.uint8),
        ("shadow_partner", np.uint8),
        ("single_byte_sig", np.uint16),
    ]
)

CYCLE_DTYPE = np.dtype(
    [
        ("micro_ref", np.uint8),
        ("micro_weight", np.uint8),
        ("step", np.uint8),
        ("phase_position", np.uint8),
        ("byte", np.uint8),
        ("intron", np.uint8),
        ("family_phase", np.uint8),
        ("state_before", np.uint32),
        ("state_after", np.uint32),
        ("u6_after", np.uint8),
        ("v6_after", np.uint8),
        ("chirality6", np.uint8),
        ("chi_shell", np.uint8),
        ("arch_shell", np.uint8),
        ("prefix_parity", np.uint8),
        ("prefix_tau_u6", np.uint8),
        ("prefix_tau_v6", np.uint8),
        ("prefix_q_transport", np.uint8),
        ("fold_disagreement", np.uint8),
        ("on_equality_horizon", np.uint8),
        ("on_complement_horizon", np.uint8),
    ]
)

DEPTH2_DTYPE = np.dtype(
    [
        ("b0", np.uint8),
        ("b1", np.uint8),
        ("s1", np.uint32),
        ("s2", np.uint32),
        ("sig_id", np.uint16),
        ("q_transport", np.uint8),
        ("sig_parity", np.uint8),
    ]
)

SIGNATURE_WORDS_DTYPE = np.dtype(
    [
        ("sig_id", np.uint16),
        ("length", np.uint8),
        ("b0", np.uint8),
        ("b1", np.uint8),
        ("b2", np.uint8),
        ("b3", np.uint8),
        ("parity", np.uint8),
        ("tau_u6", np.uint8),
        ("tau_v6", np.uint8),
        ("inverse_sig_id", np.uint16),
        ("action_on_rest", np.uint16),
    ]
)

COLLISION_DTYPE = np.dtype(
    [
        ("kind", "U16"),
        ("length_a", np.uint8),
        ("a0", np.uint8),
        ("a1", np.uint8),
        ("a2", np.uint8),
        ("a3", np.uint8),
        ("length_b", np.uint8),
        ("b0", np.uint8),
        ("b1", np.uint8),
        ("b2", np.uint8),
        ("b3", np.uint8),
        ("endpoint_a", np.uint32),
        ("endpoint_b", np.uint32),
        ("sig_a", np.uint16),
        ("sig_b", np.uint16),
    ]
)

MEASURES: dict[str, Any] = {
    "uniform": {"policy": "iid", "p_byte": "1/256"},
    "canonical": {
        "policy": "frame_sequence",
        "p_micro": "1/64",
        "family_order": [0, 1, 2, 3],
    },
    "lambda": {
        "policy": "micro_weighted",
        "p_m": "lambda^popcount(m)/(1+lambda)^6",
        "grid": [0.25, 0.5, 1, 2, 4],
    },
}


def _family_word_for_micro(micro_ref: int) -> list[int]:
    return [int(byte_from_family_micro(fam, micro_ref, D)) for fam in range(4)]


def _cycle_word_for_micro(micro_ref: int) -> list[int]:
    return _family_word_for_micro(micro_ref) * 2


def canonical_rest_swapped_rest() -> bool:
    """Depth-4/8 holonomy: every micro cycle is rest → swapped → rest."""
    for m in range(64):
        word = _cycle_word_for_micro(m)
        state = GENE_MAC_REST
        for t, byte in enumerate(word):
            state = step_state_by_byte(state, byte)
            if t == 3 and state != GENE_MAC_SWAPPED:
                return False
        if state != GENE_MAC_REST:
            return False
    return True


def build_byte_fiber() -> np.ndarray:
    rows = np.empty(256, dtype=BYTE_FIBER_DTYPE)
    for b in range(256):
        intron = intron_from_byte(b, D)
        fwd4 = intron & 0x0F
        rev4 = (intron >> 4) & 0x0F
        rows[b] = (
            b,
            intron,
            intron_family_d(intron, D),
            intron_micro_ref_d(intron, D),
            api.q_word6(b),
            api.Q_WEIGHT_BY_BYTE[b],
            fwd4,
            rev4,
            fold_disagreement_d(b, D),
            int(fwd4 == rev4),
            api.shadow_partner_byte(b),
            kernel.word_signature_id([b]),
        )
    return rows


def build_canonical_cycles() -> np.ndarray:
    rows = np.empty(64 * 8, dtype=CYCLE_DTYPE)
    i = 0
    for m in range(64):
        word = _cycle_word_for_micro(m)
        state = GENE_MAC_REST
        q_acc = 0
        for step, byte in enumerate(word):
            before = state
            state = step_state_by_byte(state, byte)
            q_acc = (q_acc ^ api.q_word6(byte)) & 0x3F
            omega = api.state24_to_omega12(state)
            chi = omega.chirality6
            chi_shell = chi.bit_count()
            intron = intron_from_byte(byte, D)
            prefix = word[: step + 1]
            sig = api.omega_word_signature(prefix)
            rows[i] = (
                m,
                int(m).bit_count(),
                step,
                step % 4,
                byte,
                intron,
                intron_family_d(intron, D),
                before,
                state,
                omega.u6,
                omega.v6,
                chi,
                chi_shell,
                6 - chi_shell,
                sig.parity,
                sig.tau_u6,
                sig.tau_v6,
                q_acc,
                fold_disagreement_d(byte, D),
                int(omega.is_on_equality_horizon),
                int(omega.is_on_complement_horizon),
            )
            i += 1
    return rows


def build_depth2_witnesses() -> np.ndarray:
    rows = np.empty(256 * 256, dtype=DEPTH2_DTYPE)
    i = 0
    for b0 in range(256):
        s1 = step_state_by_byte(GENE_MAC_REST, b0)
        for b1 in range(256):
            s2 = step_state_by_byte(s1, b1)
            word = [b0, b1]
            sig = api.omega_word_signature(word)
            rows[i] = (
                b0,
                b1,
                s1,
                s2,
                kernel.signature_id(sig.parity, sig.tau_u6, sig.tau_v6),
                (api.q_word6(b0) ^ api.q_word6(b1)) & 0x3F,
                sig.parity,
            )
            i += 1
    return rows


def _pack_word(word: bytes | list[int], max_len: int = 4) -> tuple[int, tuple[int, int, int, int]]:
    raw = [int(b) for b in word]
    length = len(raw)
    if length > max_len:
        raise ValueError(f"word length {length} exceeds max_len={max_len}")
    padded = raw + [0] * (max_len - length)
    return length, (padded[0], padded[1], padded[2], padded[3])


def _replay_word(word: bytes | list[int], start: int = GENE_MAC_REST) -> int:
    state = int(start)
    for b in word:
        state = step_state_by_byte(state, int(b))
    return state


def _word_bytes_from_row(row: np.ndarray, which: str) -> list[int]:
    length = int(row[f"length_{which}"])
    keys = [f"{which}{i}" for i in range(4)]
    return [int(row[k]) for k in keys[:length]]


def build_signature_words() -> np.ndarray:
    """All 8192 minimal representative words joined to ``signature_dataset``."""
    from src.tools.autoencoder.datasets import signature_dataset
    from src.tools.autoencoder.helpers.evals_datasets import (
        minimal_representative_words,
    )

    reps = minimal_representative_words(max_len=4)
    sig_table = signature_dataset()
    rest_omega = api.state24_to_omega12(GENE_MAC_REST)
    rows = np.empty(8192, dtype=SIGNATURE_WORDS_DTYPE)
    for sig_id in range(8192):
        word = reps[sig_id]
        length, (b0, b1, b2, b3) = _pack_word(word)
        parity, tau_u, tau_v = kernel.sig_id_parts(sig_id)
        action = int(sig_table["action_on_rest"][sig_id])
        end = _replay_word(word)
        omega = api.state24_to_omega12(end)
        end_uv = (int(omega.u6) << 6) | int(omega.v6)
        if end_uv != action:
            raise RuntimeError(
                f"signature_words replay mismatch at sig_id={sig_id}: "
                f"replay_uv={end_uv} action_on_rest={action}"
            )
        dest = api.apply_omega_signature(
            rest_omega, api.OmegaSignature12(parity, tau_u, tau_v)
        )
        dest_uv = (int(dest.u6) << 6) | int(dest.v6)
        if dest_uv != action:
            raise RuntimeError(f"action_on_rest inconsistent at sig_id={sig_id}")
        if (parity, tau_u, tau_v) != (
            int(sig_table["parity"][sig_id]),
            int(sig_table["tau_u6"][sig_id]),
            int(sig_table["tau_v6"][sig_id]),
        ):
            raise RuntimeError(f"factor mismatch at sig_id={sig_id}")
        rows[sig_id] = (
            sig_id,
            length,
            b0,
            b1,
            b2,
            b3,
            parity,
            tau_u,
            tau_v,
            int(sig_table["inverse_sig_id"][sig_id]),
            action,
        )
    return rows


def build_collisions(
    depth2: np.ndarray, signature_words: np.ndarray, *, seed: int = 0
) -> np.ndarray:
    """Deterministic provenance teaching set with fixed 4-byte storage + lengths."""
    rng = np.random.default_rng(seed)
    out: list[tuple] = []

    seen_shadow: set[tuple[int, int]] = set()
    for b in range(256):
        partner = int(api.shadow_partner_byte(b))
        pair = (min(b, partner), max(b, partner))
        if pair in seen_shadow or b == partner:
            continue
        seen_shadow.add(pair)
        ea = _replay_word([pair[0]])
        eb = _replay_word([pair[1]])
        if ea != eb:
            raise RuntimeError(f"shadow endpoints diverge for {pair}")
        out.append(
            (
                "shadow",
                1,
                pair[0],
                0,
                0,
                0,
                1,
                pair[1],
                0,
                0,
                0,
                ea,
                eb,
                kernel.word_signature_id([pair[0]]),
                kernel.word_signature_id([pair[1]]),
            )
        )

    by_sig: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(depth2):
        by_sig[int(row["sig_id"])].append(idx)

    def _sample_same_sig(*, max_pairs: int) -> None:
        keys = sorted(by_sig)
        rng.shuffle(keys)
        n = 0
        for key in keys:
            idxs = by_sig[key]
            if len(idxs) < 2:
                continue
            rng.shuffle(idxs)
            for a, b in zip(idxs[0::2], idxs[1::2]):
                ra, rb = depth2[a], depth2[b]
                wa = [int(ra["b0"]), int(ra["b1"])]
                wb = [int(rb["b0"]), int(rb["b1"])]
                ea, eb = int(ra["s2"]), int(rb["s2"])
                sa = kernel.word_signature_id(wa)
                sb = kernel.word_signature_id(wb)
                if sa != sb:
                    raise RuntimeError("same_sig pair has unequal signatures")
                out.append(
                    (
                        "same_sig",
                        2,
                        wa[0],
                        wa[1],
                        0,
                        0,
                        2,
                        wb[0],
                        wb[1],
                        0,
                        0,
                        ea,
                        eb,
                        sa,
                        sb,
                    )
                )
                n += 1
                if n >= max_pairs:
                    return

    _sample_same_sig(max_pairs=2048)

    by_end: dict[int, list[int]] = defaultdict(list)
    for sid in range(8192):
        row = signature_words[sid]
        word = [int(row[f"b{i}"]) for i in range(int(row["length"]))]
        end = _replay_word(word)
        by_end[end].append(sid)

    end_keys = sorted(by_end)
    rng.shuffle(end_keys)
    n_end = 0
    for end in end_keys:
        members = by_end[end]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        sid_a, sid_b = int(members[0]), int(members[1])
        if sid_a == sid_b:
            continue
        ra, rb = signature_words[sid_a], signature_words[sid_b]
        la = int(ra["length"])
        lb = int(rb["length"])
        wa = [int(ra[f"b{i}"]) for i in range(la)]
        wb = [int(rb[f"b{i}"]) for i in range(lb)]
        ea, eb = _replay_word(wa), _replay_word(wb)
        if ea != eb or ea != end:
            raise RuntimeError("same_end replay failed")
        if sid_a == sid_b:
            continue
        out.append(
            (
                "same_end",
                la,
                int(ra["b0"]),
                int(ra["b1"]),
                int(ra["b2"]),
                int(ra["b3"]),
                lb,
                int(rb["b0"]),
                int(rb["b1"]),
                int(rb["b2"]),
                int(rb["b3"]),
                ea,
                eb,
                sid_a,
                sid_b,
            )
        )
        n_end += 1
        if n_end >= 2048:
            break

    arr = np.empty(len(out), dtype=COLLISION_DTYPE)
    for i, row in enumerate(out):
        arr[i] = row
        # Build-time class semantics.
        kind = str(arr[i]["kind"])
        wa = _word_bytes_from_row(arr[i], "a")
        wb = _word_bytes_from_row(arr[i], "b")
        ea = _replay_word(wa)
        eb = _replay_word(wb)
        if ea != int(arr[i]["endpoint_a"]) or eb != int(arr[i]["endpoint_b"]):
            raise RuntimeError(f"collision replay mismatch at row {i}")
        sa = kernel.word_signature_id(wa)
        sb = kernel.word_signature_id(wb)
        if sa != int(arr[i]["sig_a"]) or sb != int(arr[i]["sig_b"]):
            raise RuntimeError(f"collision sig mismatch at row {i}")
        if kind == "shadow":
            if ea != eb:
                raise RuntimeError("shadow class endpoint mismatch")
        elif kind == "same_sig":
            if sa != sb or wa == wb:
                raise RuntimeError("same_sig class invariant failed")
        elif kind == "same_end":
            if ea != eb or sa == sb:
                raise RuntimeError("same_end class invariant failed")
    return arr


def run_null_invariants(
    byte_fiber: np.ndarray,
    cycles: np.ndarray,
    depth2: np.ndarray,
    collisions: np.ndarray,
    signature_words: np.ndarray | None = None,
) -> dict[str, bool]:
    """Stage-0 gates. Corpus must not be written if any fail."""
    checks: dict[str, bool] = {}

    fd_hist = np.bincount(byte_fiber["fold_disagreement"], minlength=5)
    checks["fold_disagreement_histogram"] = bool(
        np.array_equal(fd_hist, np.array([16, 64, 96, 64, 16]))
    )
    flat = byte_fiber[byte_fiber["is_flat"] == 1]
    checks["flat_byte_count"] = len(flat) == 16
    aa = byte_fiber[byte_fiber["byte"] == GENE_MIC_S][0]
    checks["archetype_flat_zero_intron"] = bool(
        aa["is_flat"] == 1 and aa["intron"] == 0
    )

    cycle_ok = True
    shell_ok = True
    mid_ok = True
    for m in range(64):
        block = cycles[cycles["micro_ref"] == m]
        if len(block) != 8:
            cycle_ok = False
            break
        if int(block[3]["state_after"]) != GENE_MAC_SWAPPED:
            cycle_ok = False
        if int(block[7]["state_after"]) != GENE_MAC_REST:
            cycle_ok = False
        k = int(m).bit_count()
        expected = [k, 6, k, 0, k, 6, k, 0]
        if [int(x) for x in block["arch_shell"]] != expected:
            shell_ok = False
        if not (
            int(block[1]["on_equality_horizon"]) == 1
            and int(block[1]["chirality6"]) == 0
        ):
            mid_ok = False
        if not (
            int(block[3]["on_complement_horizon"]) == 1
            and int(block[3]["state_after"]) == GENE_MAC_SWAPPED
        ):
            mid_ok = False
    checks["canonical_cycles_rest_swapped_rest"] = cycle_ok
    checks["arch_shell_path_template"] = shell_ok
    checks["equality_and_complement_midpoints"] = mid_ok

    finals = depth2["s2"]
    unique, counts = np.unique(finals, return_counts=True)
    checks["depth2_4096_finals"] = len(unique) == 4096
    checks["depth2_16_witnesses_each"] = bool(np.all(counts == 16))

    shadow_ok = True
    for b in range(256):
        partner = int(api.shadow_partner_byte(b))
        for idx in range(0, 4096, 64):
            s = kernel.state24_from_index(idx)
            if step_state_by_byte(s, b) != step_state_by_byte(s, partner):
                shadow_ok = False
                break
        if not shadow_ok:
            break
    checks["shadow_partner_shares_carrier_action"] = shadow_ok

    census = byte_census_arrays()
    cross = True
    for b in range(256):
        if int(byte_fiber[b]["family_phase"]) != int(census["family_u2"][b]):
            cross = False
            break
        if int(byte_fiber[b]["micro_ref"]) != int(census["micro_ref_u6"][b]):
            cross = False
            break
        if int(byte_fiber[b]["q6"]) != int(census["q6"][b]):
            cross = False
            break
    checks["byte_fiber_matches_census"] = cross

    checks["collisions_nonempty"] = len(collisions) > 0
    kinds = {str(k) for k in collisions["kind"]}
    checks["collisions_three_kinds"] = kinds >= {"shadow", "same_sig", "same_end"}

    if signature_words is not None:
        checks["signature_words_8192"] = len(signature_words) == 8192
        ids = set(int(x) for x in signature_words["sig_id"])
        checks["signature_words_cover_group"] = ids == set(range(8192))
        factors_ok = True
        for row in signature_words[::64]:
            sid = int(row["sig_id"])
            p, u, v = kernel.sig_id_parts(sid)
            if (p, u, v) != (int(row["parity"]), int(row["tau_u6"]), int(row["tau_v6"])):
                factors_ok = False
                break
        checks["signature_words_factors"] = factors_ok
    return checks


def generate_null_dataset(out_dir: Path | None = None) -> Path:
    """Build the null corpus under ``dataset_null/``. Raises if invariants fail."""
    target = Path(out_dir) if out_dir is not None else paths.dataset_dir("null")
    target.mkdir(parents=True, exist_ok=True)

    byte_fiber = build_byte_fiber()
    cycles = build_canonical_cycles()
    depth2 = build_depth2_witnesses()
    signature_words = build_signature_words()
    collisions = build_collisions(depth2, signature_words)
    checks = run_null_invariants(
        byte_fiber, cycles, depth2, collisions, signature_words
    )
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"null corpus invariants failed: {failed}")

    arrays = {
        "byte_fiber": byte_fiber,
        "canonical_cycles": cycles,
        "depth2_witnesses": depth2,
        "signature_words": signature_words,
        "collisions": collisions,
    }
    for name, arr in arrays.items():
        np.save(target / f"{name}.npy", arr)

    (target / "measures.json").write_text(
        json.dumps(MEASURES, indent=2) + "\n", encoding="utf-8"
    )

    manifest = DatasetManifest(
        dataset_name="null",
        config={
            "GENE_MIC_S": GENE_MIC_S,
            "GENE_MAC_REST": GENE_MAC_REST,
            "GENE_MAC_SWAPPED": GENE_MAC_SWAPPED,
            "finite_aperture": FINITE_APERTURE,
            "depth4_alignment": DEPTH4_ALIGNMENT,
        },
        checks=checks,
    )
    manifest.arrays = {
        name: {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_bytes": int(arr.nbytes),
        }
        for name, arr in arrays.items()
    }
    manifest.row_count = int(depth2.shape[0])
    manifest.kernel_fingerprint = DatasetManifest.kernel_fingerprint_of(
        Path(constants.__file__).parent
    )
    manifest.to_json(target / "manifest.json")
    return target


def load_null_dataset(out_dir: Path | None = None) -> dict[str, Any]:
    """Load the null corpus, generating it if absent."""
    target = Path(out_dir) if out_dir is not None else paths.dataset_dir("null")
    need = (
        not (target / "manifest.json").exists()
        or not (target / "signature_words.npy").exists()
        or "length_a" not in str(np.load(target / "collisions.npy", allow_pickle=False).dtype)
    )
    if need:
        generate_null_dataset(target)
    data: dict[str, Any] = {
        p.stem: np.load(p, allow_pickle=False)
        for p in sorted(target.glob("*.npy"))
    }
    data["measures"] = json.loads((target / "measures.json").read_text(encoding="utf-8"))
    data["manifest"] = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    return data


def micro_ref_holdout_ids(*, n_holdout: int = 16, seed: int = 0) -> np.ndarray:
    """Deterministic micro-ref holdout: popcount in {0,1,5,6}, then seed fill."""
    primary = [m for m in range(64) if int(m).bit_count() in {0, 1, 5, 6}]
    if len(primary) >= n_holdout:
        return np.asarray(sorted(primary[:n_holdout]), dtype=np.uint8)
    remaining = [m for m in range(64) if m not in primary]
    rng = np.random.default_rng(seed)
    need = n_holdout - len(primary)
    extra = rng.choice(remaining, size=need, replace=False).tolist()
    return np.asarray(sorted(primary + list(extra)), dtype=np.uint8)


def signature_is_composite(sig_id: int) -> bool:
    """Composite signatures have both tau factors nonzero."""
    _parity, tau_u, tau_v = kernel.sig_id_parts(int(sig_id))
    return tau_u != 0 and tau_v != 0


def signature_factor_holdout(sig_id: int) -> bool:
    """Hold out composites whose low bit of tau_u is set; keep all factors."""
    if not signature_is_composite(sig_id):
        return False
    _parity, tau_u, _tau_v = kernel.sig_id_parts(int(sig_id))
    return (tau_u & 1) == 1


def _ledger_key(length: int, b0: int, b1: int, b2: int, b3: int) -> tuple:
    return (int(length), int(b0), int(b1), int(b2), int(b3))


def _row_ledger_keys(row: np.ndarray) -> tuple[tuple, tuple]:
    ka = _ledger_key(
        int(row["length_a"]),
        int(row["a0"]),
        int(row["a1"]),
        int(row["a2"]),
        int(row["a3"]),
    )
    kb = _ledger_key(
        int(row["length_b"]),
        int(row["b0"]),
        int(row["b1"]),
        int(row["b2"]),
        int(row["b3"]),
    )
    return ka, kb


def assign_collision_component_splits(
    collisions: np.ndarray, *, holdout_frac: float = 0.25, seed: int = 0
) -> tuple[dict[tuple, str], np.ndarray]:
    """Assign whole connected components of ledgers to train or holdout.

    Invariant: ``train_ledgers ∩ holdout_ledgers = ∅``, and both members of a
    kept pair share the same split.
    """
    parent: dict[tuple, tuple] = {}

    def find(x: tuple) -> tuple:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple, b: tuple) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in collisions:
        ka, kb = _row_ledger_keys(row)
        union(ka, kb)

    comps: dict[tuple, list[tuple]] = defaultdict(list)
    for key in list(parent):
        comps[find(key)].append(key)

    roots = sorted(comps)
    rng = np.random.default_rng(seed)
    rng.shuffle(roots)
    n_hold = max(1, int(round(len(roots) * holdout_frac)))
    hold_roots = set(roots[:n_hold])
    ledger_split: dict[tuple, str] = {}
    for root, members in comps.items():
        split = "holdout" if root in hold_roots else "train"
        for m in members:
            ledger_split[m] = split

    keep = np.zeros(len(collisions), dtype=bool)
    for i, row in enumerate(collisions):
        ka, kb = _row_ledger_keys(row)
        if ledger_split[ka] == ledger_split[kb]:
            keep[i] = True
    return ledger_split, keep


def verify_ledger_disjoint_splits(
    collisions: np.ndarray, ledger_split: dict[tuple, str]
) -> bool:
    """True when train and holdout ledger sets are disjoint and pairs are mono-split."""
    train = {k for k, v in ledger_split.items() if v == "train"}
    hold = {k for k, v in ledger_split.items() if v == "holdout"}
    if train & hold:
        return False
    for row in collisions:
        ka, kb = _row_ledger_keys(row)
        if ledger_split.get(ka) != ledger_split.get(kb):
            return False
    return True


class NullCorpus:
    """Null corpus with structured holdouts (plan Part 3.3)."""

    def __init__(self, out_dir: Path | None = None) -> None:
        data = load_null_dataset(out_dir)
        self.byte_fiber: np.ndarray = data["byte_fiber"]
        self.cycles: np.ndarray = data["canonical_cycles"]
        self.depth2: np.ndarray = data["depth2_witnesses"]
        self.signature_words: np.ndarray = data["signature_words"]
        self.collisions: np.ndarray = data["collisions"]
        self.measures: dict[str, Any] = data["measures"]
        self.manifest: dict[str, Any] = data["manifest"]

        self.holdout_micros = micro_ref_holdout_ids(n_holdout=16, seed=0)
        self.train_micros = np.asarray(
            sorted(set(range(64)) - set(int(m) for m in self.holdout_micros)),
            dtype=np.uint8,
        )
        self.ledger_split, keep = assign_collision_component_splits(self.collisions)
        self.collisions_split = self.collisions[keep]
        if not verify_ledger_disjoint_splits(
            self.collisions_split, self.ledger_split
        ):
            raise RuntimeError("collision connected-component split invariant failed")
        if len(self.collisions_split) == 0:
            raise RuntimeError("no collision pairs survived split assignment")

        # Signature-word holdout: composites with tau_u low bit set.
        self.sig_holdout_mask = np.array(
            [signature_factor_holdout(int(s)) for s in self.signature_words["sig_id"]],
            dtype=bool,
        )
        self.sig_train = self.signature_words[~self.sig_holdout_mask]
        self.sig_holdout = self.signature_words[self.sig_holdout_mask]

    def _micros_for(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train_micros
        if split == "holdout":
            return self.holdout_micros
        raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")

    def canonical_frames(self, split: str = "train") -> np.ndarray:
        """First depth-4 frame per micro: shape [N, 4] uint8 bytes."""
        micros = self._micros_for(split)
        out = np.empty((len(micros), 4), dtype=np.uint8)
        for i, m in enumerate(micros):
            block = self.cycles[self.cycles["micro_ref"] == int(m)]
            out[i] = block["byte"][:4]
        return out

    def canonical_cycles(self, split: str = "train") -> np.ndarray:
        """Full depth-8 ledgers per micro: shape [N, 8] uint8 bytes."""
        micros = self._micros_for(split)
        out = np.empty((len(micros), 8), dtype=np.uint8)
        for i, m in enumerate(micros):
            block = self.cycles[self.cycles["micro_ref"] == int(m)]
            out[i] = block["byte"][:8]
        return out

    def cycle_rows(self, split: str = "train") -> np.ndarray:
        """Structured cycle rows restricted to the split's micros."""
        micros = set(int(m) for m in self._micros_for(split))
        mask = np.array(
            [int(r["micro_ref"]) in micros for r in self.cycles], dtype=bool
        )
        return self.cycles[mask]

    def collision_pairs(
        self, split: str = "train", kind: str | None = None
    ) -> np.ndarray:
        """Collision rows whose ledgers are assigned to ``split`` (both members)."""
        rows = []
        for row in self.collisions_split:
            k = str(row["kind"])
            if kind is not None and k != kind:
                continue
            ka, _kb = _row_ledger_keys(row)
            if self.ledger_split.get(ka) == split:
                rows.append(row)
        if not rows:
            return np.empty(0, dtype=COLLISION_DTYPE)
        out = np.empty(len(rows), dtype=COLLISION_DTYPE)
        for i, row in enumerate(rows):
            out[i] = row
        return out

    def signature_words_split(self, split: str = "train") -> np.ndarray:
        if split == "train":
            return self.sig_train
        if split == "holdout":
            return self.sig_holdout
        raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")

    def signature_holdout_mask(self, sig_ids: np.ndarray) -> np.ndarray:
        """Boolean mask: True where the signature is in the factor-holdout set."""
        return np.array(
            [signature_factor_holdout(int(s)) for s in sig_ids], dtype=bool
        )

    def incomplete_prior_frames(
        self, n: int = 256, seed: int = 0, split: str = "train"
    ) -> np.ndarray:
        """Incomplete-prior 4-byte ledgers for residual training.

        Mixes depth-2 witnesses, family-biased bytes, multi-byte corruption,
        adjacent swaps, and uniform random words. Canonical null is not used.
        ``split='holdout'`` uses a disjoint seed band from training.
        """
        from src.tools.autoencoder.helpers.evals_datasets import biased_family_bytes

        if split not in ("train", "holdout"):
            raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")
        if split == "holdout":
            seed = 9973 + int(seed)
        rng = np.random.default_rng(seed)
        out = np.empty((n, 4), dtype=np.uint8)
        d2 = self.depth2
        cycles = self.canonical_cycles("train")
        for i in range(n):
            kind = int(rng.integers(0, 5))
            if kind == 0 and len(d2):
                row = d2[int(rng.integers(0, len(d2)))]
                out[i] = (int(row["b0"]), int(row["b1"]), 0, 0)
            elif kind == 1:
                fam = int(rng.integers(0, 4))
                out[i] = biased_family_bytes(rng, fam, 4)
            elif kind == 2 and len(cycles):
                row = cycles[int(rng.integers(0, len(cycles)))][:4].copy()
                n_hit = int(rng.integers(1, 3))
                pos = rng.choice(4, size=n_hit, replace=False)
                for p in pos:
                    row[p] = int(rng.integers(0, 256))
                out[i] = row
            elif kind == 3 and len(cycles):
                row = cycles[int(rng.integers(0, len(cycles)))][:4].copy()
                a = int(rng.integers(0, 3))
                row[a], row[a + 1] = row[a + 1], row[a]
                out[i] = row
            else:
                out[i] = rng.integers(0, 256, size=4, dtype=np.uint8)
        return out

    def corruption_n_bytes(self, split: str = "train") -> int:
        """Train uses 1-byte corruption; holdout/test uses 2-byte transfer."""
        if split == "train":
            return 1
        if split == "holdout":
            return 2
        raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")

    def frame_position_indices(self, position: int) -> np.ndarray:
        """Row indices into canonical_cycles with phase_position == position."""
        if position not in (0, 1, 2, 3):
            raise ValueError(f"frame position must be 0..3, got {position}")
        return np.where(self.cycles["phase_position"] == position)[0]

    def stratify_by_frame_position(
        self, values: np.ndarray, positions: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Group a metric vector by frame position 0..3."""
        out: dict[int, np.ndarray] = {}
        for p in range(4):
            out[p] = values[positions == p]
        return out

    def stratify_by_family(
        self, values: np.ndarray, families: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Group a metric vector by family phase 0..3."""
        out: dict[int, np.ndarray] = {}
        for fam in range(4):
            out[fam] = values[families == fam]
        return out

