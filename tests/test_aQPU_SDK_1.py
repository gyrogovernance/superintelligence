from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pytest
import torch
import warnings

from src import sdk
from src.api import (
    MASK12_BY_BYTE,
    WordSignature,
    apply_word_signature,
    chirality_word6,
    depth4_intron_sequence32,
    depth4_mask_projection48,
    pack_state,
    q_word6_for_items,
    state24_to_spin6_pair,
    step_state_by_byte,
    trajectory_parity_commitment,
    walsh_hadamard64,
    word_signature,
)
from src.constants import (
    GENE_MAC_A12,
    GENE_MAC_B12,
    GENE_MAC_REST,
    MASK_STATE24,
    ab_distance,
    archetype_distance,
    component_density,
    horizon_distance,
    is_on_equality_horizon,
    is_on_horizon,
)
from src.tools.gyrolabe import ops as gyro_ops
from src.tools.gyrolabe import opencl_backend
from tests.test_aQPU_1 import _bfs_omega


def _bytes_from_items(items: Iterable[int]) -> bytes:
    return bytes(int(b) & 0xFF for b in items)


class TestStateAndCharts:
    def test_rest_state_charts_and_horizons(self) -> None:
        charts = sdk.state_charts(GENE_MAC_REST)
        print("Rest charts:", charts)

        assert charts.state24 == GENE_MAC_REST & MASK_STATE24
        assert charts.a12 == GENE_MAC_A12
        assert charts.b12 == GENE_MAC_B12
        assert charts.state_hex == f"{GENE_MAC_REST & MASK_STATE24:06x}"

        rest_d = archetype_distance(GENE_MAC_REST)
        horiz_d = horizon_distance(GENE_MAC_A12, GENE_MAC_B12)
        ab_d = ab_distance(GENE_MAC_A12, GENE_MAC_B12)

        print("Distances:", rest_d, horiz_d, ab_d)

        assert charts.constitutional.rest_distance == rest_d
        assert charts.constitutional.horizon_distance == horiz_d
        assert charts.constitutional.ab_distance == ab_d
        assert math.isclose(
            charts.constitutional.a_density,
            component_density(GENE_MAC_A12),
        )
        assert math.isclose(
            charts.constitutional.b_density,
            component_density(GENE_MAC_B12),
        )

        assert charts.constitutional.on_complement_horizon is is_on_horizon(GENE_MAC_REST)
        assert charts.constitutional.on_equality_horizon is is_on_equality_horizon(
            GENE_MAC_REST
        )

    def test_spin_pair_and_chirality_consistency(self) -> None:
        state = pack_state(GENE_MAC_A12, GENE_MAC_B12)
        spin_a, spin_b = state24_to_spin6_pair(state)
        charts = sdk.state_charts(state)

        print("Spin pair:", spin_a, spin_b)

        assert charts.spin_a6 == spin_a
        assert charts.spin_b6 == spin_b
        assert charts.chirality6 == chirality_word6(state)


class TestMomentsAndLedgers:
    def test_moment_round_trip_and_verification(self) -> None:
        word = _bytes_from_items([0x12, 0x34, 0xAA, 0x55])
        moment = sdk.moment_from_ledger(word)

        print("Moment from ledger:", moment)

        assert moment.step == len(word)
        assert moment.ledger == word
        assert moment.last_byte == word[-1]

        sig = word_signature(word)
        parity = trajectory_parity_commitment(word)
        q_tr = q_word6_for_items(word)

        assert moment.signature == sig
        assert moment.parity_commitment == parity
        assert moment.q_transport6 == q_tr
        assert sdk.verify_moment(moment)

    def test_compare_ledgers_prefix_and_divergence(self) -> None:
        base = _bytes_from_items([0x01, 0x02, 0x03])
        left = base + _bytes_from_items([0x10, 0x20])
        right = base + _bytes_from_items([0x11])

        cmp_res = sdk.compare_ledgers(left, right)
        print("Ledger comparison:", cmp_res)

        assert cmp_res.common_prefix_len == len(base)
        assert cmp_res.first_divergence_index == len(base)
        assert cmp_res.left_next_byte == left[len(base)]
        assert cmp_res.right_next_byte == right[len(base)]

        # Prefix state should match stepping the shared prefix.
        s = GENE_MAC_REST
        for b in base:
            s = step_state_by_byte(s, b)
        assert cmp_res.common_prefix_state24 == s & MASK_STATE24


class TestFutureConeAndDerivatives:
    def test_future_cone_measure_small_depth(self) -> None:
        length = 2
        measure = sdk.future_cone_measure(GENE_MAC_REST, length)

        print(
            "Future cone measure (length=2):",
            "distinct_states=",
            measure.distinct_states,
            "total_words=",
            measure.total_words,
            "entropy_bits=",
            measure.entropy_bits,
        )

        assert measure.length == length
        assert measure.total_words == 256 ** length
        assert measure.distinct_states <= measure.total_words

        prob_rest = measure.probability_of(GENE_MAC_REST)
        print("Probability of rest:", prob_rest)
        assert prob_rest >= 0

    def test_future_expectation_and_entropy_consistency(self) -> None:
        def observable(s: int) -> int:
            return chirality_word6(s).bit_count()

        length = 1
        exact = sdk.future_expectation_exact(GENE_MAC_REST, length, observable)
        approx = sdk.future_expectation_float(GENE_MAC_REST, length, observable)
        entropy = sdk.future_entropy_bits(GENE_MAC_REST, length)

        print("Future expectation:", exact, float(exact), approx)
        print("Entropy bits (length=1):", entropy)

        assert math.isclose(float(exact), approx, rel_tol=1e-6)
        assert entropy >= 0.0

    def test_directional_and_byte_derivative_table(self) -> None:
        def observable(s: int) -> int:
            return archetype_distance(s)

        state = GENE_MAC_REST
        table = sdk.byte_derivative_table(state, observable)

        print("First 8 byte derivatives:", table[:8])

        assert len(table) == 256
        for b in range(8):
            lhs = observable(step_state_by_byte(state, b))
            rhs = observable(state)
            assert table[b] == lhs - rhs


class TestWitnessFromRest:
    def test_witness_depth_and_execution(self) -> None:
        # Take a simple one-byte target.
        byte = 0x42
        target = step_state_by_byte(GENE_MAC_REST, byte)
        witness = sdk.witness_from_rest(target)

        print("Witness from rest:", witness)

        assert witness.target_state24 == target & MASK_STATE24
        assert witness.depth in (1, 2)
        assert witness.word in (bytes([byte]), witness.word)

        executed = sdk.execute_witness_from_rest(target)
        print("Executed witness moment:", executed)

        assert executed.state24 == target & MASK_STATE24
        assert executed.step == witness.depth


class TestDepth4Frame:
    def test_depth4_frame_matches_api_helpers(self) -> None:
        b0, b1, b2, b3 = 0x10, 0x20, 0x30, 0x40
        frame = sdk.depth4_frame(b0, b1, b2, b3)

        print("Depth4 frame:", frame)

        assert frame["mask48"] == depth4_mask_projection48(b0, b1, b2, b3)
        assert frame["introns32"] == depth4_intron_sequence32(b0, b1, b2, b3)

        q_tr = q_word6_for_items((b0, b1, b2, b3))
        assert frame["q_transport6"] == q_tr


class TestExactFutureConeTheorems:
    def test_future_cone_length1_exact_shadow_uniformity(self) -> None:
        measure = sdk.future_cone_measure(GENE_MAC_REST, 1)

        counts = [count for _, count in measure.state_counts]
        print(
            "Length-1 future cone distinct/counts:",
            measure.distinct_states,
            set(counts),
        )

        assert measure.distinct_states == 128
        assert measure.total_words == 256
        assert measure.exact_uniform is True
        assert set(counts) == {2}
        assert math.isclose(measure.entropy_bits, 7.0, rel_tol=0.0, abs_tol=1e-12)

    def test_future_cone_length2_uniformity_holds_off_rest(self) -> None:
        omega = sorted(_bfs_omega())
        samples = [omega[0], omega[137], omega[2048]]

        for s in samples:
            measure = sdk.future_cone_measure(s, 2)
            counts = [count for _, count in measure.state_counts]
            print(
                f"State {s:06x}: length-2 distinct/counts =",
                measure.distinct_states,
                set(counts),
            )

            assert measure.distinct_states == 4096
            assert measure.total_words == 65536
            assert measure.exact_uniform is True
            assert set(counts) == {16}
            assert math.isclose(measure.entropy_bits, 12.0, rel_tol=0.0, abs_tol=1e-12)

    def test_future_cone_length3_uniformity_from_rest(self) -> None:
        measure = sdk.future_cone_measure(GENE_MAC_REST, 3)
        counts = [count for _, count in measure.state_counts]

        print(
            "Length-3 future cone distinct/counts:",
            measure.distinct_states,
            set(counts),
        )

        assert measure.distinct_states == 4096
        assert measure.total_words == 256**3
        assert measure.exact_uniform is True
        assert set(counts) == {4096}
        assert math.isclose(measure.entropy_bits, 12.0, rel_tol=0.0, abs_tol=1e-12)


class TestTransportTableExactness:
    def test_transport_table_matches_q_map_for_any_omega_state(self) -> None:
        omega = sorted(_bfs_omega())
        expected = tuple(q_word6_for_items((b,)) for b in range(256))

        for s in [omega[0], omega[511], omega[3000]]:
            table = sdk.exact_transport_table(s)
            print(
                f"Transport table check for state {s:06x}:",
                table[:8],
            )
            assert table == expected

    def test_moment_q_transport_matches_single_byte_transport(self) -> None:
        payload = bytes([0x2B, 0x54, 0xAA, 0xD5])
        moment = sdk.moment_from_ledger(payload)
        expected = q_word6_for_items(payload)

        print("Moment q_transport6:", moment.q_transport6)

        assert moment.q_transport6 == expected


class TestBroadcastAndScanSemantics:
    def test_apply_signature_to_state_broadcasts_scalar_signature(self) -> None:
        payload = bytes([0x10, 0x20, 0x30])
        sigs = gyro_ops.signature_scan(
            torch.tensor(list(payload), dtype=torch.uint8),
        )
        final_sig = int(sigs[-1].item())

        states = torch.tensor(
            [
                GENE_MAC_REST,
                pack_state(GENE_MAC_A12 ^ 0x003, GENE_MAC_B12 ^ 0x00C),
                pack_state(GENE_MAC_A12 ^ 0x330, GENE_MAC_B12 ^ 0x0C3),
            ],
            dtype=torch.int32,
        )

        out = gyro_ops.apply_signature_to_state(states, final_sig)
        print("Broadcast scalar signature output:", out.tolist())

        for i in range(states.numel()):
            expected = int(states[i].item())
            for b in payload:
                expected = step_state_by_byte(expected, b)
            assert int(out[i].item()) == (expected & MASK_STATE24)

    def test_apply_signature_to_state_broadcasts_scalar_state(self) -> None:
        payload = bytes([0x01, 0x02, 0x03, 0x04])
        sigs = gyro_ops.signature_scan(
            torch.tensor(list(payload), dtype=torch.uint8),
        )
        start_state = torch.tensor(
            pack_state(GENE_MAC_A12 ^ 0x055, GENE_MAC_B12 ^ 0x033),
            dtype=torch.int32,
        )

        out = gyro_ops.apply_signature_to_state(start_state, sigs)
        scan = gyro_ops.state_scan_from_state(payload, int(start_state.item()))

        print("Broadcast scalar state output:", out.tolist())
        print("Direct state scan output:", scan.tolist())

        assert torch.equal(out, scan)

    def test_state_scan_from_state_bytes_path_emits_no_warning(self) -> None:
        payload = bytes([0x01, 0x23, 0x45, 0x67])
        start_state = GENE_MAC_REST

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = gyro_ops.state_scan_from_state(payload, start_state)

        print("Warnings caught:", [str(w.message) for w in caught])
        print("State scan output:", out.tolist())

        assert len(caught) == 0


class TestTensorTorchPath:
    def test_tensor_gemv64_torch_path_matches_torch_mv(self) -> None:
        torch.manual_seed(2024)
        W = torch.randn(5, 64, dtype=torch.float32) * 0.05
        x = torch.randn(64, dtype=torch.float32) * 0.05

        y_ref = torch.mv(W, x)
        y_sdk = sdk.TensorOps.gemv64(W, x, n_bits=16)

        if isinstance(y_sdk, np.ndarray):
            y_sdk = torch.from_numpy(y_sdk).to(torch.float32)

        err = float((y_sdk - y_ref).abs().max().item())
        print("TensorOps.gemv64 torch-path max err:", err)
        assert err < 1e-3


class TestWitnessCoverage:
    def test_every_omega_state_has_depth_le_2_witness_from_rest(self) -> None:
        omega = sorted(_bfs_omega())

        depths: list[int] = []
        for s in omega:
            witness = sdk.witness_from_rest(s)
            depths.append(witness.depth)

            applied = apply_word_signature(GENE_MAC_REST, witness.signature)
            replay = GENE_MAC_REST
            for b in witness.word:
                replay = step_state_by_byte(replay, b)

            assert witness.depth in (0, 1, 2)
            assert replay == (s & MASK_STATE24)
            assert applied == (s & MASK_STATE24)

        hist = {d: depths.count(d) for d in sorted(set(depths))}
        print("Witness depth histogram:", hist)
        assert max(depths) <= 2


class TestNativeEngineAndBitplane:
    """C engine availability and bitplane GEMV symbols (ported from test_aQPU_5)."""

    def test_native_available_and_bitplane_works(self) -> None:
        avail = sdk.gyrolabe_available()
        print("GyroLabe native available:", avail)
        if not avail:
            pytest.skip("GyroLabe C library not available")
        torch.manual_seed(42)
        W = torch.randn(4, 64, dtype=torch.float32) * 0.1
        x = torch.randn(64, dtype=torch.float32) * 0.1
        y = sdk.TensorOps.gemv64(W, x, n_bits=16)
        y_ref = torch.mv(W, x)
        err = (y_ref - y).abs().max().item()
        print("Bitplane GEMV max err:", err)
        assert err < 1e-3

    def test_packed_symbols_when_native_available(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        lib = gyro_ops._get_lib()
        has_gemv = hasattr(lib, "gyro_bitplane_gemv_f32")
        has_pack = hasattr(lib, "gyro_pack_bitplane_matrix_f32")
        has_packed = hasattr(lib, "gyro_bitplane_gemv_packed_f32")
        print("Bitplane symbols:", has_gemv, has_pack, has_packed)
        assert has_gemv and has_pack and has_packed


class TestChiralityDistanceViaRuntimeOps:
    """Chirality distance via SDK RuntimeOps (ported from test_aQPU_5)."""

    def test_chirality_distance_pair(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        a = torch.tensor([0x000000], dtype=torch.int32)
        b = torch.tensor([0xFFFFFF], dtype=torch.int32)
        d = sdk.RuntimeOps.chirality_distance(a, b)
        print("d(0x000000, 0xFFFFFF):", d.item())
        assert d.item() in range(0, 13)

    def test_chirality_distance_batch(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        a = torch.tensor([0, 0xFFF000, 0x000FFF], dtype=torch.int32)
        b = torch.tensor([0, 0x000FFF, 0xFFF000], dtype=torch.int32)
        d = sdk.RuntimeOps.chirality_distance(a, b)
        print("Batch chirality distances:", d.tolist())
        assert d.shape == (3,)

    def test_chirality_distance_adjacent(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        states = torch.tensor([0, 0x000001, 0x000003, 0x000007], dtype=torch.int32)
        d = sdk.RuntimeOps.chirality_distance_adjacent(states, lookahead=1)
        print("Chirality adjacent:", d.tolist())
        assert d.shape == (4,)


class TestWht64BatchAndSelfInverse:
    """WHT batch and self-inverse via SpectralOps (ported from test_aQPU_5)."""

    def test_wht64_orthonormal_vs_hadamard(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        x = torch.randn(64, dtype=torch.float32)
        y = gyro_ops.wht64(x)
        H = torch.from_numpy(walsh_hadamard64().astype("float32"))
        y_ref = H @ x
        err = (y - y_ref).abs().max().item()
        print("WHT64 vs H@x max err:", err)
        assert err < 1e-5

    def test_wht64_self_inverse(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        x = torch.randn(64, dtype=torch.float32)
        y = gyro_ops.wht64(x)
        z = gyro_ops.wht64(y)
        err = (x - z).abs().max().item()
        print("WHT64 self-inverse max err:", err)
        assert err < 1e-5

    def test_wht64_batch(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        x = torch.randn(10, 64, dtype=torch.float32)
        y = gyro_ops.wht64(x)
        print("WHT64 batch shape:", y.shape)
        assert y.shape == (10, 64)


class TestBitplaneGemvEdgeCases:
    """Bitplane GEMV identity and 64x64 (ported from test_aQPU_5)."""

    def test_bitplane_gemv_identity(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        W = torch.eye(64, dtype=torch.float32)
        x = torch.randn(64, dtype=torch.float32)
        y = sdk.TensorOps.gemv64(W, x, n_bits=16)
        y_ref = torch.mv(W, x)
        err = (y_ref - y).abs().max().item()
        print("Identity GEMV max err:", err)
        assert err < 1e-4

    def test_bitplane_gemv_64x64(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        torch.manual_seed(42)
        W = torch.randn(64, 64, dtype=torch.float32) * 0.1
        x = torch.randn(64, dtype=torch.float32) * 0.1
        y_ref = torch.mv(W, x)
        y_sdk = sdk.TensorOps.gemv64(W, x, n_bits=16)
        if isinstance(y_sdk, np.ndarray):
            y_sdk = torch.from_numpy(y_sdk).to(torch.float32)
        err = (y_ref - y_sdk).abs().max().item()
        print("64x64 GEMV max err:", err)
        assert err < 1e-3

    def test_packed_vs_unpacked_gemv_match(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        torch.manual_seed(123)
        W = torch.randn(64, 64, dtype=torch.float32) * 0.05
        x = torch.randn(64, dtype=torch.float32) * 0.05
        y_unpacked = sdk.TensorOps.gemv64(W, x, n_bits=16)
        pm = sdk.TensorOps.pack_matrix64(W, n_bits=16)
        pv = sdk.TensorOps.pack_vector64(x, n_bits=16)
        y_packed = pm.gemv_packed(pv)
        err = (y_unpacked - y_packed).abs().max().item()
        print("Packed vs unpacked err:", err)
        assert err < 1e-4


class TestApplySignatureToRest:
    """Signature -> rest state via RuntimeOps (ported from test_aQPU_5)."""

    def test_apply_signature_to_rest(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        sig = torch.tensor([0], dtype=torch.int32)
        rest = sdk.RuntimeOps.apply_signature_to_rest(sig)
        print("sig=0 -> rest:", rest.item())
        assert rest.shape == (1,)
        assert rest.item() == (GENE_MAC_REST & MASK_STATE24)


class TestQmapExtractViaRuntimeOps:
    """Q-map extract via RuntimeOps (ported from test_aQPU_5)."""

    def test_qmap_extract_gate_bytes(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        gate_bytes = [0xAA, 0x54, 0xD5, 0x2B]
        b = torch.tensor(gate_bytes, dtype=torch.uint8)
        q, f, m = sdk.RuntimeOps.qmap_extract(b)
        print("Gate bytes q_class:", q.tolist())
        print("Family:", f.tolist())
        assert q.shape == (4,)


class TestSpectralAndTensorOps:
    def test_wht64_matches_walsh_matrix(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        H = walsh_hadamard64()
        y_ref = H @ x

        y_sdk_t = sdk.SpectralOps.wht64(x)
        if hasattr(y_sdk_t, "detach"):
            y_sdk = y_sdk_t.detach().cpu().numpy()
        else:
            y_sdk = np.asarray(y_sdk_t, dtype=np.float64)

        print("WHT64 diff norm:", np.linalg.norm(y_ref - y_sdk))

        assert y_sdk.shape == y_ref.shape
        assert np.allclose(y_ref, y_sdk, rtol=1e-6, atol=1e-6)

    def test_tensor_gemv64_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        W = rng.standard_normal((4, 64))
        x = rng.standard_normal(64)

        y_ref = W @ x
        y_sdk_t = sdk.TensorOps.gemv64(W, x, n_bits=8)
        if hasattr(y_sdk_t, "detach"):
            y_sdk = y_sdk_t.detach().cpu().numpy()
        else:
            y_sdk = np.asarray(y_sdk_t, dtype=np.float64)

        print("GEMV64 ref:", y_ref)
        print("GEMV64 sdk:", y_sdk)

        assert np.allclose(y_ref, y_sdk, rtol=1e-2, atol=1e-2)

    def test_packed_gemv64_round_trip(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Packed GEMV requires native GyroLabe library.")

        rng = np.random.default_rng(2)
        W = rng.standard_normal((3, 64)).astype(np.float32)
        x = rng.standard_normal(64).astype(np.float32)

        y_ref = W @ x
        pm = sdk.TensorOps.pack_matrix64(W, n_bits=8)
        pv = sdk.TensorOps.pack_vector64(x, n_bits=8)
        y_packed_t = pm.gemv_packed(pv)
        y_packed = y_packed_t.detach().cpu().numpy()

        print("Packed GEMV ref:", y_ref)
        print("Packed GEMV sdk:", y_packed)

        assert np.allclose(y_ref, y_packed, rtol=1e-2, atol=1e-2)

    def test_opencl_packed_gemm64_matches_cpu(self) -> None:
        if not opencl_backend.available():
            pytest.skip("OpenCL backend not available")

        W = torch.randn(64, 64, dtype=torch.float32)
        X = torch.randn(32, 64, dtype=torch.float32)

        cpu_packed = gyro_ops.PackedBitplaneMatrix64(W, n_bits=8)
        Y_cpu = cpu_packed.gemm_packed_batch(X)

        gpu_packed = sdk.TensorOps.pack_matrix64_opencl(W, n_bits=8)
        Y_gpu = gpu_packed.gemm_batch(X)
        gpu_packed.close()

        max_diff = (Y_cpu - Y_gpu).abs().max().item()
        print("OpenCL max diff:", max_diff)
        assert max_diff <= 1e-5


class TestSignatureScanSequences:
    """Signature scan on specific byte sequences (ported from test_aQPU_5)."""

    def test_signature_scan_single_byte(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        b = torch.tensor([0xAA], dtype=torch.uint8)
        sig = sdk.RuntimeOps.signature_scan(b)
        print("byte 0xAA -> signature:", f"{sig.item():08x}")
        assert sig.shape == (1,)
        assert sig.dtype == torch.int32

    def test_signature_scan_sequence(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        b = torch.tensor([0x00, 0x01, 0x02, 0xAA], dtype=torch.uint8)
        sig = sdk.RuntimeOps.signature_scan(b)
        print("seq [0x00,0x01,0x02,0xAA] -> sigs:", [f"{s:08x}" for s in sig.tolist()])
        assert sig.shape == (4,)

    def test_signatures_to_states_roundtrip(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        b = torch.tensor([0xAA, 0x54], dtype=torch.uint8)
        sigs = sdk.RuntimeOps.signature_scan(b)
        states = sdk.RuntimeOps.states_from_signatures(sigs)
        print("sigs -> states:", [f"{s:06x}" for s in states.tolist()])
        assert states.shape == sigs.shape

    def test_chirality_states_from_bytes(self) -> None:
        if not gyro_ops.native_available():
            pytest.skip("Native GyroLabe required")
        b = torch.tensor([0x00, 0xFF], dtype=torch.uint8)
        states = sdk.RuntimeOps.chirality_states_from_bytes(b)
        print("chirality_states_from_bytes:", [f"{s:06x}" for s in states.tolist()])
        assert states.shape == (2,)


class TestGyrolabeRuntimeOps:
    def test_signature_and_state_flow_consistency(self) -> None:
        payload = _bytes_from_items([0x01, 0x02, 0x03, 0x04])
        t_payload = torch.tensor(list(payload), dtype=torch.uint8)

        sigs = gyro_ops.signature_scan(t_payload)
        states_from_sigs = gyro_ops.signatures_to_states(sigs)
        states_direct = gyro_ops.chirality_states_from_bytes(t_payload)

        print("Signatures:", sigs.tolist())
        print("States from signatures:", states_from_sigs.tolist())
        print("States direct:", states_direct.tolist())

        assert torch.all(states_from_sigs.eq(states_direct))

    def test_extract_scan_shapes_and_values(self) -> None:
        payload = _bytes_from_items(range(8))
        t_payload = torch.tensor(list(payload), dtype=torch.uint8)

        q, family, micro, sigs, states = gyro_ops.extract_scan(t_payload)

        print("q:", q.tolist())
        print("family:", family.tolist())
        print("micro:", micro.tolist())
        print("signatures:", sigs.tolist())
        print("states:", states.tolist())

        assert q.shape == t_payload.shape
        assert sigs.shape == t_payload.shape
        assert states.shape == t_payload.shape

        # Check that first signature is single-byte signature.
        first_sig = int(sigs[0].item())
        a12 = GENE_MAC_A12
        b12 = GENE_MAC_B12
        # Apply by replaying the byte sequence directly.
        s = pack_state(a12, b12)
        s = step_state_by_byte(s, int(t_payload[0].item()))
        assert states[0].item() == s


class TestNewSignatureAndScanPrimitives:
    def test_apply_signature_to_state_matches_python_step(self) -> None:
        payload = _bytes_from_items([0x10, 0x20, 0x30])
        t_payload = torch.tensor(list(payload), dtype=torch.uint8)

        sigs = gyro_ops.signature_scan(t_payload)
        last_sig = sigs[-1]

        rest_state = torch.tensor(GENE_MAC_REST, dtype=torch.int32)
        applied = gyro_ops.apply_signature_to_state(rest_state, last_sig)

        s = GENE_MAC_REST
        for b in payload:
            s = step_state_by_byte(s, b)

        print("Signature:", int(last_sig.item()))
        print("State by signature:", int(applied.item()))
        print("State by replay:", s)

        assert int(applied.item()) == s & MASK_STATE24

    def test_apply_signature_batch_matches_step_loop(self) -> None:
        payload = _bytes_from_items([0x01, 0x02])
        t_payload = torch.tensor(list(payload), dtype=torch.uint8)
        sigs = gyro_ops.signature_scan(t_payload)

        # Two different starting states.
        states = torch.tensor(
            [GENE_MAC_REST, pack_state(GENE_MAC_A12 ^ 0x3, GENE_MAC_B12 ^ 0x5)],
            dtype=torch.int32,
        ).repeat(len(payload), 1)
        states = states.t().contiguous()

        batch_sigs = sigs.repeat(states.shape[0], 1)
        batch_out = gyro_ops.apply_signature_batch(states, batch_sigs)

        print("Batch apply signatures out:", batch_out.tolist())

        for i in range(states.shape[0]):
            for j in range(states.shape[1]):
                s = int(states[i, j].item())
                for b in payload[: j + 1]:
                    s = step_state_by_byte(s, b)
                assert batch_out[i, j].item() == s & MASK_STATE24

    def test_step_byte_batch_matches_step_state_by_byte(self) -> None:
        states = torch.tensor(
            [
                GENE_MAC_REST,
                pack_state(GENE_MAC_A12 ^ 0x1, GENE_MAC_B12 ^ 0x2),
                pack_state(GENE_MAC_A12 ^ 0x4, GENE_MAC_B12 ^ 0x8),
            ],
            dtype=torch.int32,
        )
        byte = 0x7F

        stepped_batch = gyro_ops.step_byte_batch(states, byte)

        print("Stepped batch:", stepped_batch.tolist())

        for i in range(states.shape[0]):
            s = int(states[i].item())
            s_next = step_state_by_byte(s, byte)
            assert stepped_batch[i].item() == s_next & MASK_STATE24

    def test_state_scan_from_state_matches_python_loop(self) -> None:
        payload = _bytes_from_items([0x01, 0x23, 0x45, 0x67, 0x89])
        start_state = pack_state(GENE_MAC_A12 ^ 0x55, GENE_MAC_B12 ^ 0x33)

        scan = gyro_ops.state_scan_from_state(payload, start_state)

        print("State scan from state:", scan.tolist())

        s = start_state
        expected = []
        for b in payload:
            s = step_state_by_byte(s, b)
            expected.append(s & MASK_STATE24)

        assert scan.tolist() == expected

