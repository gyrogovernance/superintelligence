"""Dataset schemas, census builders, dense tables, and versioned manifests.

Everything here exists to generate and load the exact kernel-derived datasets
that land under data/ (gitignored, deterministically regenerable via the CLI's
``generate`` subcommand). All values come from the hQVM kernel; this module
only reshapes kernel calls into arrays and records the schema + invariant
checks in a JSON manifest per dataset. Signature id packing lives in
``kernel.signature_id`` / ``kernel.signature_from_id``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src import api, constants
from src.family import fold_disagreement_d

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
    return table


def inverse_transition_table() -> np.ndarray:
    """inverse_state[4096, 256] -> canonical index of the predecessor."""
    # Cached: the 4096 x 256 kernel build is ~1M calls, and many readouts,
    # datasets, and replay routines reuse it. The inverse is a pure kernel
    # function of state + byte, so a single cached instance is correct.
    return _INVERSE_TRANSITION_TABLE_CACHE


@lru_cache(maxsize=1)
def _build_inverse_transition_table() -> np.ndarray:
    table = np.empty((4096, 256), dtype=np.uint16)
    for index in range(4096):
        state24 = int(kernel.state24_from_index(index))
        for byte in range(256):
            table[index, byte] = kernel.state_index(
                constants.inverse_step_by_byte(state24, byte)
            )
    return table


_INVERSE_TRANSITION_TABLE_CACHE = _build_inverse_transition_table()


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
    (the repo layout ``data/dataset_<name>/``)."""
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
        "bytes", "states", "transitions", "actions", "signatures"
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