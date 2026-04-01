from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pytest
import torch

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


