"""Tests for exact group actions used by equivariant models (spec 4.4, 4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from src import api

from src.tools.autoencoder import kernel


def test_state_index_roundtrip_all_states() -> None:
    for index in range(4096):
        state24 = int(kernel.state24_from_index(index))
        assert kernel.state_index(state24) == index


def test_step_index_matches_kernel_all_states_sampled_bytes() -> None:
    for index in range(0, 4096, 53):
        for byte in (0x00, 0x11, 0xAA, 0x54, 0xD5, 0xFF):
            stepped = kernel.step_index(index, byte)
            state24 = int(kernel.state24_from_index(index))
            dest = api.step_state_by_byte(state24, byte)
            assert stepped == kernel.state_index(dest)


def test_apply_signature_index_matches_kernel() -> None:
    for sig_id in (0, 1, 63, 64, 4131, 8191):
        for index in (0, 1, 100, 2089, 4095):
            via_adapter = kernel.apply_signature_index(index, sig_id)
            parity, tau_u6, tau_v6 = kernel.sig_id_parts(sig_id)
            omega = api.OmegaState12(u6=(index >> 6) & 63, v6=index & 63)
            dest = api.apply_omega_signature(
                omega, api.OmegaSignature12(parity, tau_u6, tau_v6)
            )
            assert via_adapter == ((dest.u6 << 6) | dest.v6)


def test_signature_ids_enumerate_group() -> None:
    ids = kernel.enumerate_signature_ids()
    assert np.array_equal(ids, np.arange(8192))
    # all 8192 ids decode to distinct (parity, tau_u, tau_v)
    decoded = {kernel.sig_id_parts(int(i)) for i in ids}
    assert len(decoded) == 8192


def test_k4_orbit_of_indices_is_consistent() -> None:
    # orbit closure under the adapter's gate application
    for index in range(0, 4096, 211):
        orbit = {index}
        for gate in ("S", "C", "F"):
            orbit.add(kernel.apply_k4_index(index, gate))
        for member in orbit:
            for gate in ("S", "C", "F"):
                assert kernel.apply_k4_index(member, gate) in orbit
        # orbit ids from the kernel match
        state24 = int(kernel.state24_from_index(index))
        kernel_orbit = api.k4_orbit(state24)
        adapter_orbit = {int(kernel.state24_from_index(m)) for m in orbit}
        assert kernel_orbit == frozenset(adapter_orbit)


def test_word_signature_id_matches_replay() -> None:
    words = [(0x12,), (0x12, 0x34), (0xD5, 0x2B, 0x7E, 0x11)]
    for word in words:
        sig_id = kernel.word_signature_id(word)
        parity, tau_u6, tau_v6 = kernel.sig_id_parts(sig_id)
        # replay from state index 0 must land on the signature's action
        rest = api.OmegaState12(u6=0, v6=0)  # state index 0, valid Omega member
        replayed = rest
        for byte in word:
            replayed = api.step_omega12_by_byte(replayed, byte)
        applied = api.apply_omega_signature(
            rest, api.OmegaSignature12(parity, tau_u6, tau_v6)
        )
        assert (replayed.u6, replayed.v6) == (applied.u6, applied.v6)
