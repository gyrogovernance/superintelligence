"""Tests for dataset generation (spec 4)."""

from __future__ import annotations

import numpy as np
import pytest

from src import api, constants
from src.family import fold_disagreement_d

from src.tools.autoencoder import datasets, kernel
from src.tools.autoencoder.helpers.evals_datasets import (
    minimal_representative_words,
)
from src.tools.autoencoder.datasets import DatasetManifest


@pytest.fixture(scope="module")
def byte_census():
    return datasets.byte_census_arrays()


@pytest.fixture(scope="module")
def state_census():
    return datasets.state_census_arrays()


@pytest.fixture(scope="module")
def transitions():
    return {"next_state": datasets.transition_table()}


# ---------------------------------------------------------------------------
# Byte census
# ---------------------------------------------------------------------------


def test_byte_census_shape_and_fields(byte_census) -> None:
    for name, arr in byte_census.items():
        assert arr.shape == (256,), name


def test_byte_census_matches_kernel_tables(byte_census) -> None:
    assert np.array_equal(byte_census["q6"], np.array([api.q_word6(b) for b in range(256)]))
    assert np.array_equal(byte_census["family_u2"], np.array(api.FAMILY_BY_BYTE))
    assert np.array_equal(byte_census["micro_ref_u6"], np.array(api.MICRO_REF_BY_BYTE))
    # shadow pair ids are 0..127 and both members share an id
    ids = byte_census["shadow_pair_id"]
    for byte in range(256):
        partner = api.shadow_partner_byte(byte)
        assert ids[byte] == ids[partner] == min(byte, partner)
    assert len(set(ids.tolist())) == 128


def test_byte_census_family_decomposition(byte_census) -> None:
    assert np.array_equal(
        byte_census["family_u2"],
        byte_census["family_bit7"] * 2 + byte_census["family_bit0"],
    )


def test_byte_census_full_column_set(byte_census) -> None:
    # Dataset-contract check: every census column the audit and downstream
    # codecs read must match the kernel's public per-byte tables. This pins the
    # AE's byte census to the kernel tables it was built from; it does not
    # re-prove any kernel theorem.
    assert np.array_equal(byte_census["mask12"], np.array(api.MASK12_BY_BYTE))
    assert np.array_equal(byte_census["intron_u8"], np.array(api.INTRON_BY_BYTE))
    assert np.array_equal(byte_census["q_weight"], np.array(api.Q_WEIGHT_BY_BYTE))
    assert np.array_equal(
        byte_census["fold_disagreement"],
        np.array([fold_disagreement_d(b, 6) for b in range(256)]),
    )


def test_chirality_chart_matches_bit_ops(state_census) -> None:
    # The spectral model, codecs, ensembles, and metrics all derive chirality
    # by bit ops (u ^ v); the census stores the kernel's chirality. This pins
    # the two representations together so a divergence would fail loudly rather
    # than silently corrupting every codec and the spectral story.
    idx = state_census["state_index"].astype(np.int64)
    assert np.array_equal(
        state_census["chirality6"].astype(np.int64), (idx >> 6) ^ (idx & 63)
    )


def test_stepping_paths_agree(transitions) -> None:
    # The AE consumes the kernel two ways: transition_table() is built via
    # api.step_omega12_by_byte; kernel.step_index goes via
    # constants.step_state_by_byte. This guards cross-consistency between those
    # two consumption paths (word-replay labels vs _embed_bytes).
    table = transitions["next_state"]
    for i in range(0, 4096, 211):
        for b in (0x00, 0x54, 0xAA, 0xD5, 0xFF):
            assert kernel.step_index(i, b) == int(table[i, b])


def test_transport_law_via_ae_artifacts(state_census, transitions) -> None:
    # The transport law chi' = chi ^ q under a byte step is the foundation of
    # the codec ladder and the Walsh story. Asserted here only through AE
    # objects: the census chirality and the AE's own transition table.
    chi = state_census["chirality6"].astype(np.int64)
    q6 = datasets.byte_census_arrays()["q6"].astype(np.int64)
    nxt = transitions["next_state"]
    for b in (0x00, 0x2A, 0x54, 0xAA, 0xD5, 0xFF):
        assert np.array_equal(chi[nxt[:, b]], chi ^ q6[b])


def test_signature_apply_matches_word_replay() -> None:
    # Optional fifth: the apply direction of the signature law, via AE
    # artifacts only. For each signature's minimal representative word (replay
    # path), applying the signature directly to rest must agree. Citations
    # #50.
    rest = api.state24_to_omega12(constants.GENE_MAC_REST)
    rest_idx = (rest.u6 << 6) | rest.v6
    reps = minimal_representative_words(max_len=4)
    for sig_id, word in reps.items():
        applied = kernel.apply_signature_index(rest_idx, sig_id)
        replayed = rest_idx
        for byte in word:  # word replay is left-to-right (first byte acts first)
            replayed = kernel.step_index(replayed, byte)
        assert applied == replayed, sig_id


# ---------------------------------------------------------------------------
# State census
# ---------------------------------------------------------------------------


def test_state_census_shape(state_census) -> None:
    for name, arr in state_census.items():
        assert arr.shape == (4096,), name


def test_state_census_index_is_chart(state_census) -> None:
    assert np.array_equal(
        state_census["state_index"],
        (state_census["u6"].astype(np.uint16) << 6) | state_census["v6"],
    )
    assert np.array_equal(state_census["state_index"], np.arange(4096))


def test_state_census_shell_fields(state_census) -> None:
    # shell_chi = popcount(chirality), arch_shell = 6 - shell_chi
    chi = state_census["chirality6"]
    assert np.array_equal(state_census["shell_chi"], _popcount(chi))
    assert np.array_equal(state_census["arch_shell"], 6 - state_census["shell_chi"])
    # exact optical fractions agree with shell
    shell = state_census["shell_chi"].astype(np.int64)
    assert np.array_equal(state_census["optical_eq_num"], shell)
    assert np.array_equal(state_census["optical_mu_num"], 2 * shell - 6)


def _popcount(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.int64)
    for bit in range(6):
        out += (arr.astype(np.int64) >> bit) & 1
    return out


def test_state_census_matches_kernel_observables(state_census) -> None:
    # spot-check 64 states across the space against direct kernel calls
    for index in range(0, 4096, 64):
        state24 = int(state_census["state24"][index])
        omega = api.state24_to_omega12(state24)
        assert state24 == int(kernel.state24_from_index(index))
        assert state_census["shell_chi"][index] == omega.shell
        assert state_census["equality_horizon"][index] == int(omega.is_on_equality_horizon)
        assert state_census["complement_horizon"][index] == int(
            omega.is_on_complement_horizon
        )
        assert state_census["a12"][index] == constants.unpack_state(state24)[0]


def test_state_census_k4_orbits_partition(state_census) -> None:
    orbit_ids = state_census["k4_orbit_id"]
    sizes = state_census["k4_orbit_size"]
    for oid in np.unique(orbit_ids):
        members = np.nonzero(orbit_ids == oid)[0]
        assert sizes[members[0]] == len(members)
    # orbit sizes are 1, 2, or 4 (K4 stabilizers)
    assert set(np.unique(sizes).tolist()) <= {1, 2, 4}


def test_state_census_spin_roundtrip(state_census) -> None:
    for index in range(0, 4096, 97):
        state24 = int(state_census["state24"][index])
        spin_a, spin_b = api.state24_to_spin6_pair(state24)
        packed_a = datasets._pack_spins6(spin_a)
        packed_b = datasets._pack_spins6(spin_b)
        assert packed_a == int(state_census["spin_a6"][index])
        assert packed_b == int(state_census["spin_b6"][index])


# ---------------------------------------------------------------------------
# Transition tables
# ---------------------------------------------------------------------------


def test_transition_table_matches_kernel(transitions) -> None:
    table = transitions["next_state"]
    for index in range(0, 4096, 37):
        omega = api.OmegaState12(u6=(index >> 6) & 63, v6=index & 63)
        for byte in range(0, 256, 41):
            dest = api.step_omega12_by_byte(omega, byte)
            assert table[index, byte] == (dest.u6 << 6) | dest.v6


def test_transition_table_inverse_roundtrip() -> None:
    nxt = datasets.transition_table()
    inv = datasets.inverse_transition_table()
    idx = np.arange(4096, dtype=np.uint16)[:, None]
    byt = np.arange(256, dtype=np.uint16)[None, :]
    expected = np.broadcast_to(idx, (4096, 256))
    assert np.array_equal(inv[nxt[idx, byt], byt], expected)


# ---------------------------------------------------------------------------
# K4 actions and signatures
# ---------------------------------------------------------------------------


def test_k4_action_arrays() -> None:
    action, fixed = kernel.k4_action_arrays()
    assert action.shape == (4, 4096)
    assert np.array_equal(action[0], np.arange(4096))
    for gate_i in (1, 2, 3):
        assert np.array_equal(action[gate_i][action[gate_i]], np.arange(4096))
        assert np.array_equal(fixed[gate_i].astype(bool), action[gate_i] == np.arange(4096))
    # verified kernel structure: S fixes equality horizon (u==v), C fixes
    # complement horizon (chi==63), F fixes nothing
    equality_indices = np.nonzero(
        np.array([api.state24_to_omega12(kernel.state24_from_index(i)).is_on_equality_horizon
                  for i in range(4096)])
    )[0]
    assert set(np.nonzero(fixed[1])[0].tolist()) == set(equality_indices.tolist())
    assert int(fixed[3].sum()) == 0


def test_signature_dataset_values() -> None:
    sigs = datasets.signature_dataset()
    assert sigs["sig_id"].shape == (8192,)
    assert np.array_equal(sigs["sig_id"], np.arange(8192))
    for sig_id in (0, 1, 64, 4096, 8191):
        parity, tau_u6, tau_v6 = kernel.sig_id_parts(sig_id)
        assert sigs["parity"][sig_id] == parity
        assert sigs["inverse_sig_id"][sig_id] == kernel.signature_inverse_id(sig_id)
    # inverse of inverse is itself
    inv = sigs["inverse_sig_id"]
    assert np.array_equal(inv[inv], np.arange(8192, dtype=np.uint16))


def test_signature_action_on_rest_matches_kernel() -> None:
    sigs = datasets.signature_dataset()
    rest = api.state24_to_omega12(constants.GENE_MAC_REST)
    for sig_id in range(0, 8192, 173):
        parity, tau_u6, tau_v6 = kernel.sig_id_parts(sig_id)
        dest = api.apply_omega_signature(rest, api.OmegaSignature12(parity, tau_u6, tau_v6))
        assert sigs["action_on_rest"][sig_id] == ((dest.u6 << 6) | dest.v6)
    # every signature's action_on_rest must be a valid state index
    assert int(sigs["action_on_rest"].max()) < 4096


# ---------------------------------------------------------------------------
# Manifests and persistence
# ---------------------------------------------------------------------------


def test_generated_dataset_saves_with_manifest(tmp_path) -> None:
    out = datasets.generate_dataset("actions", data_dir=tmp_path)
    manifest_file = out / "manifest.json"
    assert manifest_file.exists()
    arrays = datasets.load_dataset("actions", data_dir=tmp_path)
    assert set(arrays) == {"action", "fixed"}
    assert arrays["action"].shape == (4, 4096)


def test_run_invariant_checks_report_failures(tmp_path) -> None:
    bad = {"action": np.zeros((4, 4096), dtype=np.uint16),
           "fixed": np.zeros((4, 4096), dtype=np.uint8)}
    checks = datasets.run_invariant_checks("actions", bad)
    assert checks["id_identity"] is False


def test_entry_bytes_matches_nbytes() -> None:
    """entry_bytes must give the real on-disk size for every dtype in use."""
    cases = [
        (np.uint8, 256),
        (np.uint16, 4096),
        (np.uint32, 8192),
        (np.int8, 512),
        (np.float32, 4096),
        (np.float64, 1024),
        (np.int64, 1024),
    ]
    for dtype, n in cases:
        arr = np.zeros(n, dtype=dtype)
        assert DatasetManifest.entry_bytes(
            list(arr.shape), str(arr.dtype)
        ) == int(arr.nbytes), dtype


def test_generate_is_deterministic(tmp_path) -> None:
    """Generating the full dataset set twice must produce byte-identical
    arrays and manifests - the datasets are deterministic functions of the
    kernel, so any drift between runs is a real regression."""
    import filecmp
    import shutil

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    datasets.generate_all(data_dir=dir_a)
    # regenerate from a clean directory so no cached .npy is reused
    datasets.generate_all(data_dir=dir_b)

    # every emitted .npy and manifest.json must match byte-for-byte
    a_files = sorted(p for p in dir_a.rglob("*") if p.is_file())
    b_files = sorted(p for p in dir_b.rglob("*") if p.is_file())
    assert [p.name for p in a_files] == [p.name for p in b_files]
    for fa, fb in zip(a_files, b_files):
        # tolerate only the timestamp-free content; manifests carry no time
        assert fa.read_bytes() == fb.read_bytes(), fa.name
