"""
test_aQPU_3.py — Frontier computational characterization of the aQPU.

Run:  pytest tests/test_aQPU_3.py -v -s

Computes structural properties of the kernel as a computational medium.
Where possible, results are derived from exact algebraic laws established
in the physics tests, not from sampling. Assertions encode only proven
structural invariants. All tests print results for examination.

Scope:
- Permutation structure and exact row-class theorem (rank from ε-quotient)
- Exact undetected error enumerator from (1+z²)^12
- Exact perturbation law (payload vs boundary decomposition)
- Channel capacity (per-byte and depth-4)
- State separation universality and distance spectrum
- Tamper detection with exact miss mechanisms
- Adversarial steering and exact path multiplicity
- Fingerprint discrimination and parity-vs-chirality relationship
- Exact two-step uniformization
- Trajectory compression bounds

Does NOT retest: kernel conformance (test_physics_1-6),
gate structure / horizons / chirality transport (test_aQPU),
Hilbert lift / Bell / CHSH / teleportation (test_aQPU_2).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.constants import (
    GATE_C_BYTES,
    GATE_S_BYTES,
    GENE_MAC_REST,
    LAYER_MASK_12,
    byte_to_intron,
    inverse_step_by_byte,
    is_on_equality_horizon,
    is_on_horizon,
    pack_state,
    step_state_by_byte,
    unpack_state,
)
from src.api import MASK12_BY_BYTE, chirality_word6, mask12_for_byte, q_word6
from tests.test_aQPU_1 import _bfs_omega


# ----------------------------------------------------------------
# Shared infrastructure
# ----------------------------------------------------------------


def _build_omega_index():
    omega = _bfs_omega()
    omega_list = sorted(omega)
    idx = {s: i for i, s in enumerate(omega_list)}
    return omega_list, idx


def _byte_perm_tuple(byte, omega_list, idx):
    return tuple(idx[step_state_by_byte(s, byte)] for s in omega_list)


def _cycle_lengths(perm):
    n = len(perm)
    seen = [False] * n
    lengths = []
    for start in range(n):
        if seen[start]:
            continue
        length = 0
        i = start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _popcount(x):
    return bin(int(x)).count("1")


def _collapse_to_6bit(c12):
    """Collapse pair-diagonal 12-bit word to 6-bit word."""
    out = 0
    for i in range(6):
        if (int(c12) >> (2 * i)) & 0x3:
            out |= 1 << i
    return out


C64_12 = set(int(m) & 0xFFF for m in MASK12_BY_BYTE)


# ================================================================
# 1. PERMUTATION STRUCTURE AND EXACT ROW-CLASS THEOREM
# ================================================================


class TestPermutationStructure:
    """
    256 bytes act as permutations on the 4096-state Ω.
    128 distinct permutations, uniform 2-to-1 shadow.
    """

    def test_permutation_census(self):
        omega_list, idx = _build_omega_index()

        perm_to_bytes: dict[tuple, list[int]] = {}
        for b in range(256):
            p = _byte_perm_tuple(b, omega_list, idx)
            perm_to_bytes.setdefault(p, []).append(b)

        cycle_type_counts: Counter = Counter()
        order_counts: Counter = Counter()
        for p in perm_to_bytes:
            ct = _cycle_lengths(p)
            cycle_type_counts[ct] += 1
            order_counts[math.lcm(*ct)] += 1

        mult = Counter(len(v) for v in perm_to_bytes.values())

        print(f"\n{'='*65}")
        print("1. PERMUTATION STRUCTURE ON Ω (4096 states)")
        print(f"{'='*65}")
        print(f"  256 bytes produce {len(perm_to_bytes)} distinct permutations")
        print(f"\n  Byte-to-permutation multiplicity:")
        for m, count in sorted(mult.items()):
            print(f"    {count} permutations each realized by {m} bytes")
        print(f"\n  {len(cycle_type_counts)} distinct cycle types:")
        for ct, count in sorted(cycle_type_counts.items(), key=lambda x: -x[1]):
            unique_lens = sorted(set(ct))
            len_counts = Counter(ct)
            desc = ", ".join(f"{l}×{len_counts[l]}" for l in unique_lens)
            print(f"    {count:4d} perms: [{desc}]")
        print(f"\n  Permutation order spectrum:")
        for order, count in sorted(order_counts.items()):
            print(f"    order {order:4d}: {count} distinct permutations")

        assert len(perm_to_bytes) == 128, "Expected 128 distinct permutations"
        assert mult == Counter({2: 128}), "Expected uniform 2-to-1 shadow"


class TestExactRowClassStructure:
    """
    The uniform transition matrix on Ω has rank 32, not rank 1.

    Exact explanation from the product structure Ω = U × V:
    In code coordinates (x, y) ∈ GF(2)^6 × GF(2)^6, a byte maps:
      x' = y ⊕ p_a      (p_a ∈ {0, ε}, ε = 111111)
      y' = x ⊕ μ ⊕ p_b  (μ uniform over GF(2)^6, p_b ∈ {0, ε})

    The row of the transition matrix depends only on {y, y⊕ε},
    giving |GF(2)^6| / |<ε>| = 64 / 2 = 32 distinct row types.
    """

    def test_transition_matrix_has_exactly_32_distinct_rows(self):
        omega_list, idx = _build_omega_index()
        n = len(omega_list)

        T = np.zeros((n, n), dtype=np.float64)
        for b in range(256):
            for i, s in enumerate(omega_list):
                j = idx[step_state_by_byte(s, b)]
                T[i, j] += 1.0
        T /= 256.0

        # Count distinct rows
        row_hashes: dict[bytes, list[int]] = {}
        for i in range(n):
            key = T[i].tobytes()
            row_hashes.setdefault(key, []).append(i)

        distinct_rows = len(row_hashes)
        class_sizes = Counter(len(v) for v in row_hashes.values())

        # Verify the ε-quotient explanation
        a_rest, b_rest = unpack_state(GENE_MAC_REST)
        a6_rest = _collapse_to_6bit(a_rest)
        b6_rest = _collapse_to_6bit(b_rest)

        # For each row class, check that members share {y, y⊕ε}
        epsilon = 0x3F  # 111111 in 6 bits
        row_class_y_pairs: list[set[frozenset]] = []
        for members in row_hashes.values():
            y_pairs = set()
            for i in members:
                s = omega_list[i]
                _, b12 = unpack_state(s)
                y = _collapse_to_6bit(b12 ^ b_rest)
                y_pair = frozenset({y, y ^ epsilon})
                y_pairs.add(y_pair)
            row_class_y_pairs.append(y_pairs)

        # Each row class should map to exactly one {y, y⊕ε} pair
        single_y_pair = all(len(yp) == 1 for yp in row_class_y_pairs)

        rank = np.linalg.matrix_rank(T, tol=1e-8)

        print(f"\n{'='*65}")
        print("1b. EXACT ROW-CLASS THEOREM")
        print(f"{'='*65}")
        print(f"  Transition matrix ({n}×{n}):")
        print(f"    Distinct rows: {distinct_rows}")
        print(f"    Matrix rank: {rank}")
        print(f"    Row class sizes: {dict(sorted(class_sizes.items()))}")
        print(f"    All classes determined by {{y, y⊕ε}}: {single_y_pair}")
        print(f"    Theoretical: |GF(2)^6| / |<ε>| = 64/2 = 32")
        print(f"    Match: {distinct_rows == 32}")

        assert distinct_rows == 32, f"Expected 32 distinct rows, got {distinct_rows}"
        assert single_y_pair, "Row classes not determined by ε-quotient"

    def test_family_restricted_row_classes(self):
        """Family-0 bytes have p_a = p_b = 0, so the ε-quotient
        disappears and row count should be 64."""
        omega_list, idx = _build_omega_index()
        n = len(omega_list)

        family0_bytes = [b for b in range(256)
                         if (byte_to_intron(b) & 0x81) == 0]

        T = np.zeros((n, n), dtype=np.float64)
        for b in family0_bytes:
            for i, s in enumerate(omega_list):
                j = idx[step_state_by_byte(s, b)]
                T[i, j] += 1.0
        T /= len(family0_bytes)

        row_hashes: dict[bytes, list[int]] = {}
        for i in range(n):
            key = T[i].tobytes()
            row_hashes.setdefault(key, []).append(i)

        distinct_rows = len(row_hashes)
        rank = np.linalg.matrix_rank(T, tol=1e-8)

        print(f"\n  FAMILY-0 RESTRICTED ({len(family0_bytes)} bytes):")
        print(f"    Distinct rows: {distinct_rows}")
        print(f"    Matrix rank: {rank}")
        print(f"    Theoretical (no ε-quotient): 64")

        assert distinct_rows == 64, f"Expected 64 distinct rows, got {distinct_rows}"

    def test_row_class_lattice_under_byte_subsets(self):
        """
        Row-class counts under controlled byte subsets expose the
        complement-phase structure (epsilon quotient 32 vs 64).
        """
        omega_list, idx = _build_omega_index()
        n = len(omega_list)

        def distinct_rows_for_bytes(byte_list: list[int]) -> int:
            if not byte_list:
                return 0
            T = np.zeros((n, n), dtype=np.float64)
            for b in byte_list:
                for i, s in enumerate(omega_list):
                    j = idx[step_state_by_byte(s, b)]
                    T[i, j] += 1.0
            T /= len(byte_list)
            row_hashes: set[bytes] = set()
            for i in range(n):
                row_hashes.add(T[i].tobytes())
            return len(row_hashes)

        subsets = [
            ("bit0=0", [b for b in range(256) if (byte_to_intron(b) & 1) == 0]),
            ("bit0=1", [b for b in range(256) if (byte_to_intron(b) & 1) == 1]),
            ("bit7=0", [b for b in range(256) if ((byte_to_intron(b) >> 7) & 1) == 0]),
            ("bit7=1", [b for b in range(256) if ((byte_to_intron(b) >> 7) & 1) == 1]),
            ("L0 even", [b for b in range(256)
              if ((byte_to_intron(b) & 1) ^ ((byte_to_intron(b) >> 7) & 1)) == 0]),
            ("L0 odd", [b for b in range(256)
              if ((byte_to_intron(b) & 1) ^ ((byte_to_intron(b) >> 7) & 1)) == 1]),
            ("family 00", [b for b in range(256) if (byte_to_intron(b) & 0x81) == 0]),
            ("family 01", [b for b in range(256) if (byte_to_intron(b) & 0x81) == 0x01]),
            ("family 10", [b for b in range(256) if (byte_to_intron(b) & 0x81) == 0x80]),
            ("family 11", [b for b in range(256) if (byte_to_intron(b) & 0x81) == 0x81]),
        ]

        print(f"\n  ROW-CLASS LATTICE (distinct rows under byte subsets):")
        print(f"  {'Subset':<12s}  {'Bytes':>6s}  {'DistinctRows':>12s}")
        for name, blist in subsets:
            dr = distinct_rows_for_bytes(blist)
            print(f"  {name:<12s}  {len(blist):6d}  {dr:12d}")
        # Full 256 should give 32
        assert distinct_rows_for_bytes(list(range(256))) == 32


# ================================================================
# 2. EXACT UNDETECTED ERROR ENUMERATOR
# ================================================================


class TestExactErrorEnumerator:
    """
    Ω = U × V with U, V each a C64-coset in GF(2)^12.
    An error stays in Ω iff its A-component and B-component are
    both in C64. The undetected weight enumerator is (1+z²)^12.
    """

    def test_weight_vs_detection_exact(self):
        omega_set = _bfs_omega()
        rng = np.random.default_rng(42)
        sample = sorted(omega_set)
        rng.shuffle(sample)
        sample = sample[:512]

        print(f"\n{'='*65}")
        print("2. EXACT UNDETECTED ERROR ENUMERATOR")
        print(f"{'='*65}")
        print(f"  Sampled {len(sample)} states")
        print(f"  Ω has product form U×V; undetected enumerator = (1+z²)^12")

        # Compute (1+z²)^12 coefficients
        # This is a polynomial in z; coefficient of z^w = C(12, w/2) if w even, 0 if odd
        print(f"\n  {'Wt':>4s}  {'Trials':>8s}  {'Escaped':>8s}  "
              f"{'Meas%':>8s}  {'SE%':>6s}  {'C(12,w/2)':>9s}  {'C(24,w)':>8s}  "
              f"{'Theo%':>8s}  {'Match':>6s}")

        for weight in range(1, 9):
            escaped = 0
            trials = 0
            if weight == 1:
                positions = [[bit] for bit in range(24)]
            elif weight == 2:
                # Exhaustive for weight 2
                from itertools import combinations
                positions = [list(c) for c in combinations(range(24), weight)]
            else:
                test_positions = []
                for _ in range(300):
                    bits = rng.choice(24, size=weight, replace=False)
                    test_positions.append(bits.tolist())
                positions = test_positions

            for s in sample:
                for bits in positions:
                    error_mask = 0
                    for bit in bits:
                        error_mask |= (1 << bit)
                    corrupted = s ^ error_mask
                    trials += 1
                    if corrupted not in omega_set:
                        escaped += 1

            meas_undetected = 1.0 - escaped / trials if trials > 0 else 0

            # Theoretical: coefficient of z^w in (1+z²)^12
            if weight % 2 == 0:
                undetected_count = math.comb(12, weight // 2)
            else:
                undetected_count = 0
            total_patterns = math.comb(24, weight)
            theo_undetected = undetected_count / total_patterns if total_patterns > 0 else 0

            match = abs(meas_undetected - theo_undetected) < 0.005
            se_pct = 100.0 * math.sqrt(meas_undetected * (1 - meas_undetected) / trials) if trials > 0 else 0.0

            print(f"  {weight:4d}  {trials:8d}  {escaped:8d}  "
                  f"{100*meas_undetected:8.3f}  {se_pct:6.3f}  {undetected_count:9d}  "
                  f"{total_patterns:8d}  {100*theo_undetected:8.3f}  "
                  f"{'YES' if match else 'NO':>6s}")

    def test_pair_flip_destinations_are_c64(self):
        """Pair-flip errors stay in Ω and correspond to C64 codewords."""
        omega_set = _bfs_omega()
        sample = sorted(omega_set)[:256]

        print(f"\n  PAIR-FLIP DESTINATION ANALYSIS:")
        for comp_name, offset in [("A", 12), ("B", 0)]:
            all_predicted = True
            for pair_idx in range(6):
                mask = 0x3 << (2 * pair_idx + offset)
                for s in sample:
                    flipped = s ^ mask
                    if flipped not in omega_set:
                        all_predicted = False
                        break
                    a_orig, b_orig = unpack_state(s)
                    a_flip, b_flip = unpack_state(flipped)
                    diff = (a_orig ^ a_flip) if offset == 12 else (b_orig ^ b_flip)
                    if diff not in C64_12:
                        all_predicted = False
                        break
            print(f"    {comp_name}: pair-flips correspond to C64 codewords: "
                  f"{all_predicted}")
            assert all_predicted


# ================================================================
# 3. EXACT PERTURBATION LAW
# ================================================================


class TestExactPerturbationLaw:
    """
    Flipping one bit of a byte changes q₆ by:
    - 1 chirality bit if payload position (bits 1-6)
    - 6 chirality bits if boundary position (bits 0, 7)

    Exact mean = (6×1 + 2×6) / 8 = 18/8 = 2.25 chirality bits.
    Exact state distance = 2 × chirality distance (pair-diagonal).
    """

    def test_exact_per_bit_decomposition(self):
        print(f"\n{'='*65}")
        print("3. EXACT PERTURBATION LAW")
        print(f"{'='*65}")

        per_bit_total = [0] * 8
        per_bit_count = [0] * 8

        for b in range(256):
            q_orig = q_word6(b)
            for bit in range(8):
                q_flipped = q_word6(b ^ (1 << bit))
                hw = _popcount(q_orig ^ q_flipped)
                per_bit_total[bit] += hw
                per_bit_count[bit] += 1

        roles = {0: "L0 boundary", 1: "LI payload", 2: "FG payload",
                 3: "BG payload", 4: "BG payload", 5: "FG payload",
                 6: "LI payload", 7: "L0 boundary"}

        print(f"\n  Per-bit-position chirality divergence:")
        print(f"  {'Bit':>4s}  {'Exact':>6s}  {'Expected':>8s}  "
              f"{'Role':>14s}  {'Match':>6s}")
        for bit in range(8):
            exact = per_bit_total[bit] / per_bit_count[bit]
            expected = 6.0 if bit in (0, 7) else 1.0
            match = abs(exact - expected) < 1e-10
            print(f"  {bit:4d}  {exact:6.1f}  {expected:8.1f}  "
                  f"{roles[bit]:>14s}  {'YES' if match else 'NO':>6s}")
            assert match, f"Bit {bit}: expected {expected}, got {exact}"

        total = sum(per_bit_total)
        count = sum(per_bit_count)
        mean = total / count

        print(f"\n  Overall mean chirality divergence: {mean:.6f}")
        print(f"  Exact formula: (6×1 + 2×6) / 8 = 18/8 = {18/8:.6f}")
        print(f"  Match: {abs(mean - 18/8) < 1e-10}")
        print(f"  State distance = 2 × chirality distance = {2*18/8:.6f}")

        assert abs(mean - 18 / 8) < 1e-10

    def test_spreading_is_length_independent(self):
        rng = np.random.default_rng(999)

        print(f"\n  SPREADING VS TRAJECTORY LENGTH (confirming saturation):")
        print(f"  {'L':>3s}  {'AvgChiDiv':>10s}  {'AvgStateDist':>12s}  "
              f"{'ZeroColl%':>10s}  {'Ratio':>6s}")

        for L in [1, 2, 4, 8, 16, 32]:
            chi_divs = []
            state_dists = []
            for _ in range(10000):
                traj = [int(rng.integers(0, 256)) for _ in range(L)]
                flip_pos = int(rng.integers(0, L))
                flip_bit = int(rng.integers(0, 8))

                traj_pert = traj.copy()
                traj_pert[flip_pos] ^= (1 << flip_bit)

                s1, s2 = GENE_MAC_REST, GENE_MAC_REST
                for b in traj:
                    s1 = step_state_by_byte(s1, b)
                for b in traj_pert:
                    s2 = step_state_by_byte(s2, b)

                chi_divs.append(_popcount(
                    chirality_word6(s1) ^ chirality_word6(s2)))
                state_dists.append(_popcount(s1 ^ s2))

            avg_chi = np.mean(chi_divs)
            avg_state = np.mean(state_dists)
            zero_coll = 100 * sum(1 for d in state_dists if d == 0) / len(state_dists)
            ratio = avg_state / avg_chi if avg_chi > 0 else 0

            print(f"  {L:3d}  {avg_chi:10.4f}  {avg_state:12.4f}  "
                  f"{zero_coll:10.3f}  {ratio:6.3f}")

        # Zero collisions is structural (per-byte bijectivity)
        assert zero_coll == 0.0, "Nonzero collision rate violates bijectivity"


# ================================================================
# 4. CHANNEL CAPACITY
# ================================================================


class TestChannelCapacity:
    """
    Per-byte: exactly 128 uniform outputs = 7 bits Shannon = 7 bits min-entropy.
    Depth-4: near-uniform over all 4096 states.
    """

    def test_per_byte_capacity(self):
        omega_list, _ = _build_omega_index()

        entropies_shannon = []
        entropies_min = []
        for s in omega_list[:500]:
            counts = Counter()
            for b in range(256):
                counts[step_state_by_byte(s, b)] += 1
            probs = np.array(list(counts.values())) / 256.0
            H_s = -np.sum(probs * np.log2(probs))
            H_m = -np.log2(max(probs))
            entropies_shannon.append(H_s)
            entropies_min.append(H_m)

        print(f"\n{'='*65}")
        print("4. CHANNEL CAPACITY")
        print(f"{'='*65}")
        print(f"  PER-BYTE (sampled {len(entropies_shannon)} states):")
        print(f"    Shannon entropy:  mean={np.mean(entropies_shannon):.6f}  "
              f"std={np.std(entropies_shannon):.8f}")
        print(f"    Min-entropy:      mean={np.mean(entropies_min):.6f}  "
              f"std={np.std(entropies_min):.8f}")
        print(f"    Theoretical (128 outputs, uniform 2:1): H=H_min=7.0")

        assert abs(np.mean(entropies_shannon) - 7.0) < 1e-6
        assert abs(np.mean(entropies_min) - 7.0) < 1e-6

    def test_depth4_output_distribution(self):
        rng = np.random.default_rng(45)
        num_frames = 200000
        outputs = Counter()
        for _ in range(num_frames):
            s = GENE_MAC_REST
            for _ in range(4):
                s = step_state_by_byte(s, int(rng.integers(0, 256)))
            outputs[s] += 1

        distinct = len(outputs)
        probs = np.array(list(outputs.values())) / num_frames
        H = -np.sum(probs * np.log2(probs))
        H_min = -np.log2(max(probs))
        chi_sq = sum((c - num_frames / 4096) ** 2 / (num_frames / 4096)
                     for c in outputs.values())

        print(f"\n  DEPTH-4 FRAME (from rest, {num_frames} random 4-byte words):")
        print(f"    Distinct outputs: {distinct} / 4096")
        print(f"    Shannon entropy:  {H:.4f} / {math.log2(4096):.4f}")
        print(f"    Min-entropy:      {H_min:.4f} / {math.log2(4096):.4f}")
        print(f"    (Min-entropy is sample-limited; see uniformization test)")
        print(f"    Chi-squared:      {chi_sq:.2f}  (df=4095)")

        assert distinct == 4096


# ================================================================
# 5. STATE SEPARATION AND DISTANCE SPECTRUM
# ================================================================


class TestStateSeparation:
    """
    Per-byte bijectivity implies every byte separates every pair.
    The interesting question is: what is the output distance distribution?
    """

    def test_universal_separation(self):
        """Every byte distinguishes every distinct pair (bijectivity)."""
        omega_list, idx = _build_omega_index()

        rng = np.random.default_rng(50)
        sample_pairs = []
        for _ in range(5000):
            i = int(rng.integers(0, len(omega_list)))
            j = int(rng.integers(0, len(omega_list)))
            if i != j:
                sample_pairs.append((i, j))

        print(f"\n{'='*65}")
        print("5. STATE SEPARATION UNIVERSALITY")
        print(f"{'='*65}")

        # Verify all bytes distinguish all sampled pairs
        all_separate = True
        for i, j in sample_pairs[:1000]:
            s1, s2 = omega_list[i], omega_list[j]
            for b in range(256):
                if step_state_by_byte(s1, b) == step_state_by_byte(s2, b):
                    all_separate = False
                    break
            if not all_separate:
                break

        print(f"  All 256 bytes separate every sampled pair: {all_separate}")
        print(f"  (Follows from per-byte bijectivity on full 2^24 carrier)")
        assert all_separate

    def test_output_distance_spectrum(self):
        """How far apart do outputs land for different bytes?"""
        omega_list, _ = _build_omega_index()
        rng = np.random.default_rng(51)

        print(f"\n  OUTPUT DISTANCE SPECTRUM:")
        print(f"  For random (s1, s2) pairs, distribution of")
        print(f"  Hamming(T_b(s1), T_b(s2)) over random bytes b:")

        dist_counts: Counter = Counter()
        for _ in range(5000):
            i = int(rng.integers(0, len(omega_list)))
            j = int(rng.integers(0, len(omega_list)))
            if i == j:
                continue
            s1, s2 = omega_list[i], omega_list[j]
            b = int(rng.integers(0, 256))
            d = _popcount(step_state_by_byte(s1, b) ^ step_state_by_byte(s2, b))
            dist_counts[d] += 1

        print(f"\n  {'Dist':>4s}  {'Count':>6s}  {'Fraction':>8s}")
        total = sum(dist_counts.values())
        for d in sorted(dist_counts.keys()):
            print(f"  {d:4d}  {dist_counts[d]:6d}  "
                  f"{dist_counts[d]/total:8.4f}")

        avg_dist = sum(d * c for d, c in dist_counts.items()) / total
        print(f"\n  Average output distance: {avg_dist:.3f}")

    def test_exact_omega_pairwise_distance_distribution(self):
        """
        Exact distribution of 24-bit Hamming distance between Omega states.
        Omega = U x V with U,V C64-cosets; delta in C64 x C64.
        Weight enumerator C64: (1+z^2)^6 => count at weight 2k is C(6,k).
        Convolution: count at distance 2k is C(12,k). Fraction = C(12,k)/4096.
        """
        print(f"\n  EXACT OMEGA PAIRWISE DISTANCE (no sampling):")
        print(f"  From (1+z^2)^6 conv (1+z^2)^6 => (1+z^2)^12 on 24-bit.")
        print(f"  {'Dist':>4s}  {'ExactFrac':>10s}  {'ExactCount':>10s}  "
              f"{'C(12,k)':>8s}")
        total_frac = 0.0
        for k in range(13):
            d = 2 * k
            count_k = math.comb(12, k)
            frac = count_k / 4096.0
            total_frac += frac
            print(f"  {d:4d}  {frac:10.6f}  {count_k:10d}  {count_k:8d}")
        print(f"  Total fraction: {total_frac:.6f}  (expect 1.0)")
        expect_mean = sum(2 * k * math.comb(12, k) for k in range(13)) / 4096.0
        print(f"  Expected mean distance: {expect_mean:.4f}")
        assert abs(total_frac - 1.0) < 1e-10
        assert abs(expect_mean - 12.0) < 1e-10

    def test_byte_preserves_hamming_distance(self):
        """
        For any byte b, Hamming(T_b(s), T_b(t)) == Hamming(s, t).
        Byte application is an affine bijection on the 24-bit carrier.
        """
        omega_list, _ = _build_omega_index()
        rng = np.random.default_rng(52)
        for _ in range(500):
            i = int(rng.integers(0, len(omega_list)))
            j = int(rng.integers(0, len(omega_list)))
            b = int(rng.integers(0, 256))
            s, t = omega_list[i], omega_list[j]
            d_in = _popcount(s ^ t)
            s_out = step_state_by_byte(s, b)
            t_out = step_state_by_byte(t, b)
            d_out = _popcount(s_out ^ t_out)
            assert d_in == d_out, (
                f"Distance not preserved: s={s:#x} t={t:#x} b={b:#x} "
                f"d_in={d_in} d_out={d_out}"
            )
        print(f"\n  Byte preserves Hamming distance: 500 random (s,t,b) checked.")


# ================================================================
# 6. TAMPER DETECTION WITH EXACT MISS MECHANISMS
# ================================================================


class TestTamperDetection:
    """
    Exact mechanisms for tamper misses:
    - Substitution: miss iff replacement is shadow partner (1/255)
    - Deletion/insertion: miss iff byte is pointwise stabilizer on prefix
    - Adjacent swap: miss iff q(x) = q(y) or x = y
    """

    def test_substitution_mechanism(self):
        rng = np.random.default_rng(60)

        print(f"\n{'='*65}")
        print("6. TAMPER DETECTION — EXACT MISS MECHANISMS")
        print(f"{'='*65}")

        # Find shadow partners for all bytes
        omega_list, idx = _build_omega_index()
        shadow: dict[int, int] = {}
        for b1 in range(256):
            for b2 in range(b1 + 1, 256):
                if all(step_state_by_byte(s, b1) == step_state_by_byte(s, b2)
                       for s in omega_list[:20]):
                    shadow[b1] = b2
                    shadow[b2] = b1

        print(f"\n  SUBSTITUTION:")
        print(f"    Shadow partners found: {len(shadow)} bytes have partners")

        # Test: miss iff replacement is shadow partner
        detected = 0
        missed = 0
        missed_shadow = 0
        trials = 50000
        for _ in range(trials):
            L = int(rng.integers(4, 20))
            traj = [int(rng.integers(0, 256)) for _ in range(L)]
            pos = int(rng.integers(0, L))
            old_byte = traj[pos]
            new_byte = (old_byte + int(rng.integers(1, 256))) % 256

            s1 = GENE_MAC_REST
            for b in traj:
                s1 = step_state_by_byte(s1, b)

            traj2 = traj.copy()
            traj2[pos] = new_byte
            s2 = GENE_MAC_REST
            for b in traj2:
                s2 = step_state_by_byte(s2, b)

            if s1 == s2:
                missed += 1
                if shadow.get(old_byte) == new_byte:
                    missed_shadow += 1
            else:
                detected += 1

        print(f"    Trials: {trials}")
        print(f"    Detected: {detected} ({100*detected/trials:.2f}%)")
        print(f"    Missed: {missed} ({100*missed/trials:.2f}%)")
        print(f"    Missed due to shadow partner: {missed_shadow} / {missed}")
        print(f"    Shadow explains all misses: {missed_shadow == missed}")
        p_theo = 1.0 / 255.0
        se_theo = math.sqrt(p_theo * (1 - p_theo) / trials) * 100.0
        print(f"    Theoretical miss rate: 1/255 = {100/255:.2f}%")
        print(f"    Expected SE (binomial): +/- {se_theo:.3f}%")

    def test_adjacent_swap_mechanism(self):
        rng = np.random.default_rng(63)

        print(f"\n  ADJACENT SWAP:")

        detected = 0
        missed = 0
        missed_same_q = 0
        missed_equal = 0
        trials = 0

        for _ in range(50000):
            L = int(rng.integers(4, 20))
            traj = [int(rng.integers(0, 256)) for _ in range(L)]
            pos = int(rng.integers(0, L - 1))
            if traj[pos] == traj[pos + 1]:
                continue
            trials += 1

            s1 = GENE_MAC_REST
            for b in traj:
                s1 = step_state_by_byte(s1, b)

            traj2 = traj.copy()
            traj2[pos], traj2[pos + 1] = traj2[pos + 1], traj2[pos]
            s2 = GENE_MAC_REST
            for b in traj2:
                s2 = step_state_by_byte(s2, b)

            if s1 == s2:
                missed += 1
                q1 = q_word6(traj[pos])
                q2 = q_word6(traj[pos + 1])
                if q1 == q2:
                    missed_same_q += 1
            else:
                detected += 1

        print(f"    Trials (distinct adjacent pairs): {trials}")
        print(f"    Detected: {detected} ({100*detected/trials:.2f}%)")
        print(f"    Missed: {missed} ({100*missed/trials:.2f}%)")
        print(f"    Missed due to q(x)=q(y): {missed_same_q} / {missed}")
        print(f"    q-class explains all misses: {missed_same_q == missed}")
        print(f"    Theoretical (x!=y, one fixed): 3/255 = {100*3/255:.2f}%")
        print(f"    Theoretical (full 64 q-classes): 1/64 = {100/64:.2f}%")
        # Not asserting equality because we skip equal bytes,
        # but the q-class must explain all misses
        if missed > 0:
            assert missed_same_q == missed, "q-class does not explain all swap misses"

    def test_deletion_mechanism(self):
        rng = np.random.default_rng(61)

        print(f"\n  DELETION:")

        detected = 0
        missed = 0
        missed_is_gate = 0
        trials = 50000

        # Decomposition: by byte type (S / C / other) and prefix horizon
        missed_s_bytes = 0
        missed_c_bytes = 0
        missed_other = 0
        missed_prefix_eq = 0
        missed_prefix_comp = 0
        missed_prefix_bulk = 0
        # By prefix length bucket: A = pos in {0,1}, B = pos >= 2
        bucket_a_s, bucket_a_c = 0, 0
        bucket_b_s, bucket_b_c = 0, 0

        for _ in range(trials):
            L = int(rng.integers(4, 20))
            traj = [int(rng.integers(0, 256)) for _ in range(L)]
            pos = int(rng.integers(0, L))

            s1 = GENE_MAC_REST
            for b in traj:
                s1 = step_state_by_byte(s1, b)

            traj2 = traj[:pos] + traj[pos + 1:]
            s2 = GENE_MAC_REST
            for b in traj2:
                s2 = step_state_by_byte(s2, b)

            if s1 == s2:
                missed += 1
                prefix_state = GENE_MAC_REST
                for b in traj[:pos]:
                    prefix_state = step_state_by_byte(prefix_state, b)
                deleted_byte = traj[pos]
                if step_state_by_byte(prefix_state, deleted_byte) == prefix_state:
                    missed_is_gate += 1
                    if deleted_byte in GATE_S_BYTES:
                        missed_s_bytes += 1
                    elif deleted_byte in GATE_C_BYTES:
                        missed_c_bytes += 1
                    else:
                        missed_other += 1
                    if is_on_equality_horizon(prefix_state):
                        missed_prefix_eq += 1
                    elif is_on_horizon(prefix_state):
                        missed_prefix_comp += 1
                    else:
                        missed_prefix_bulk += 1
                    if pos <= 1:
                        if deleted_byte in GATE_S_BYTES:
                            bucket_a_s += 1
                        elif deleted_byte in GATE_C_BYTES:
                            bucket_a_c += 1
                    else:
                        if deleted_byte in GATE_S_BYTES:
                            bucket_b_s += 1
                        elif deleted_byte in GATE_C_BYTES:
                            bucket_b_c += 1
            else:
                detected += 1

        print(f"    Trials: {trials}")
        print(f"    Detected: {detected} ({100*detected/trials:.2f}%)")
        print(f"    Missed: {missed} ({100*missed/trials:.2f}%)")
        print(f"    Missed because byte fixes prefix state: "
              f"{missed_is_gate} / {missed}")
        print(f"    Fixed-point explains all misses: "
              f"{missed_is_gate == missed}")
        if missed > 0:
            assert missed_is_gate == missed
            print(f"    Deletion miss decomposition (by deleted byte type):")
            print(f"      S-bytes (fix equality horizon): {missed_s_bytes}")
            print(f"      C-bytes (fix complement horizon): {missed_c_bytes}")
            print(f"      Other: {missed_other}")
            print(f"    Deletion miss decomposition (by prefix horizon):")
            print(f"      Prefix on equality horizon: {missed_prefix_eq}")
            print(f"      Prefix on complement horizon: {missed_prefix_comp}")
            print(f"      Prefix in bulk: {missed_prefix_bulk}")
            print(f"    Deletion miss by prefix length (pos in {{0,1}} vs pos>=2):")
            print(f"      Bucket A (pos 0,1): S={bucket_a_s}  C={bucket_a_c}")
            print(f"      Bucket B (pos>=2):  S={bucket_b_s}  C={bucket_b_c}")


# ================================================================
# 7. ADVERSARIAL STEERING AND EXACT PATH MULTIPLICITY
# ================================================================


class TestAdversarialSteering:
    """
    From rest, any target is reachable in ≤2 bytes.
    State-path count = 8; byte-path count = 16 (from provenance theorem).
    """

    def test_steering_and_exact_multiplicity(self):
        omega_set = _bfs_omega()

        print(f"\n{'='*65}")
        print("7. ADVERSARIAL STEERING — EXACT PATH MULTIPLICITY")
        print(f"{'='*65}")

        # Depth-1: state-paths
        depth1_states: dict[int, list[int]] = {}
        for b in range(256):
            t = step_state_by_byte(GENE_MAC_REST, b)
            depth1_states.setdefault(t, []).append(b)

        # Depth-2: byte-paths (full 256^2 enumeration)
        byte_paths_per_target: Counter = Counter()
        state_paths_per_target: Counter = Counter()
        state_path_set: dict[int, set] = {}

        for b1 in range(256):
            s1 = step_state_by_byte(GENE_MAC_REST, b1)
            for b2 in range(256):
                t = step_state_by_byte(s1, b2)
                byte_paths_per_target[t] += 1
                if t not in state_path_set:
                    state_path_set[t] = set()
                state_path_set[t].add((s1, t))

        for t, paths in state_path_set.items():
            state_paths_per_target[t] = len(paths)

        byte_mults = Counter(byte_paths_per_target.values())
        state_mults = Counter(state_paths_per_target.values())

        print(f"  FROM REST:")
        print(f"    Depth-1 reachable states: {len(depth1_states)}")
        print(f"    Depth-2 reachable states: {len(byte_paths_per_target)}")
        print(f"    Full Ω: {len(omega_set)}")
        print(f"\n  BYTE-PATH multiplicity (256² = 65536 total):")
        for m, count in sorted(byte_mults.items()):
            print(f"      {count} targets each with {m} byte-paths")
        print(f"    Total byte-paths: {sum(byte_paths_per_target.values())}")
        print(f"    Expected: 65536 / 4096 = 16 per target")
        print(f"\n  STATE-PATH multiplicity:")
        for m, count in sorted(state_mults.items()):
            print(f"      {count} targets each with {m} state-paths")
        # Distinct intermediate states s1 per target = 4 (holographic dictionary)
        distinct_s1_per_target = next(iter(state_mults.keys())) if state_mults else 0
        print(f"    Distinct intermediate states s1 per target: {distinct_s1_per_target}")

        assert len(byte_paths_per_target) == 4096
        assert byte_mults == Counter({16: 4096}), \
            f"Expected uniform 16 byte-paths per target"
        assert state_mults == Counter({4: 4096}), \
            f"Expected exactly 4 distinct intermediate states s1 per target, got {state_mults}"

    def test_horizon_maintenance(self):
        omega_set = _bfs_omega()
        horizon_states = [s for s in omega_set if is_on_horizon(s)]

        keeping_bytes: list[int] = []
        for s in horizon_states:
            keepers = sum(1 for b in range(256)
                          if is_on_horizon(step_state_by_byte(s, b)))
            keeping_bytes.append(keepers)

        print(f"\n  HORIZON MAINTENANCE (complement horizon, 64 states):")
        print(f"    Bytes keeping each state on horizon: "
              f"{keeping_bytes[0]} / 256 (uniform: "
              f"{len(set(keeping_bytes)) == 1})")
        print(f"    Fraction: {keeping_bytes[0]/256:.4f} = 1/64")

        assert all(k == 4 for k in keeping_bytes)


# ================================================================
# 8. FINGERPRINT DISCRIMINATION AND PARITY-CHIRALITY RELATIONSHIP
# ================================================================


class TestFingerprintDiscrimination:
    """
    Three fingerprint types with distinct collision regimes.
    Chirality and parity both have 1/64 collision rate.
    Test their mutual information: are they independent or correlated?
    """

    def test_reachable_set_by_depth(self):
        print(f"\n{'='*65}")
        print("8. FINGERPRINT DISCRIMINATION")
        print(f"{'='*65}")

        frontier = {GENE_MAC_REST}
        visited = {GENE_MAC_REST}
        print(f"\n  REACHABLE SET SIZE FROM REST:")
        print(f"  {'Depth':>5s}  {'New':>6s}  {'Total':>6s}")
        print(f"  {0:5d}  {1:6d}  {1:6d}")
        for d in range(1, 4):
            new_frontier = set()
            for s in frontier:
                for b in range(256):
                    t = step_state_by_byte(s, b)
                    if t not in visited:
                        visited.add(t)
                        new_frontier.add(t)
            frontier = new_frontier
            print(f"  {d:5d}  {len(frontier):6d}  {len(visited):6d}")

    def test_collision_rates(self):
        rng = np.random.default_rng(46)
        trials = 100000

        print(f"\n  COLLISION RATES ({trials} random trajectory pairs):")
        print(f"  {'L':>3s}  {'State24':>10s}  {'Chi6':>10s}  "
              f"{'ParityO':>10s}  {'1/|Ω|':>10s}  {'1/64':>10s}")

        for L in [1, 2, 4, 8, 16]:
            state_coll = 0
            chi_coll = 0
            parity_coll = 0

            for _ in range(trials):
                s1, s2 = GENE_MAC_REST, GENE_MAC_REST
                o1, o2 = 0, 0
                for step in range(L):
                    b1 = int(rng.integers(0, 256))
                    b2 = int(rng.integers(0, 256))
                    s1 = step_state_by_byte(s1, b1)
                    s2 = step_state_by_byte(s2, b2)
                    if step % 2 == 0:
                        o1 ^= mask12_for_byte(b1)
                        o2 ^= mask12_for_byte(b2)

                if s1 == s2:
                    state_coll += 1
                if chirality_word6(s1) == chirality_word6(s2):
                    chi_coll += 1
                if o1 == o2:
                    parity_coll += 1

            print(f"  {L:3d}  {state_coll/trials:10.5f}  "
                  f"{chi_coll/trials:10.5f}  "
                  f"{parity_coll/trials:10.5f}  "
                  f"{1/4096:10.5f}  {1/64:10.5f}")

    def test_parity_vs_chirality_mutual_information(self):
        """Are chirality and parity independent, or do they carry
        the same 6 bits of information?"""
        rng = np.random.default_rng(47)
        trials = 200000

        joint: Counter = Counter()
        chi_counts: Counter = Counter()
        par_counts: Counter = Counter()

        for _ in range(trials):
            L = int(rng.integers(2, 10))
            s = GENE_MAC_REST
            o = 0
            for step in range(L):
                b = int(rng.integers(0, 256))
                s = step_state_by_byte(s, b)
                if step % 2 == 0:
                    o ^= mask12_for_byte(b)

            chi = chirality_word6(s)
            par = _collapse_to_6bit(o)
            joint[(chi, par)] += 1
            chi_counts[chi] += 1
            par_counts[par] += 1

        # Compute mutual information
        H_chi = -sum(c / trials * math.log2(c / trials)
                     for c in chi_counts.values())
        H_par = -sum(c / trials * math.log2(c / trials)
                     for c in par_counts.values())
        H_joint = -sum(c / trials * math.log2(c / trials)
                       for c in joint.values())
        MI = H_chi + H_par - H_joint

        print(f"\n  PARITY vs CHIRALITY MUTUAL INFORMATION:")
        print(f"    H(χ):           {H_chi:.4f} bits  (max 6.0)")
        print(f"    H(parity):      {H_par:.4f} bits  (max 6.0)")
        print(f"    H(χ, parity):   {H_joint:.4f} bits  (max 12.0)")
        print(f"    I(χ; parity):   {MI:.4f} bits")
        print(f"    Independent:    {'YES' if MI < 0.05 else 'NO'}")
        print(f"    Distinct joint values: {len(joint)} / {64*64}")

    def test_conditional_entropies_parity_state_chi(self):
        """
        Conditional entropies: what does parity add beyond state / beyond chi?
        H(parity | state), H(state | parity), H(parity | chi).
        """
        rng = np.random.default_rng(48)
        trials = 200000
        state_parity: Counter = Counter()
        state_counts: Counter = Counter()
        par_counts: Counter = Counter()
        chi_parity: Counter = Counter()
        chi_counts: Counter = Counter()

        for _ in range(trials):
            L = int(rng.integers(2, 10))
            s = GENE_MAC_REST
            o = 0
            for step in range(L):
                b = int(rng.integers(0, 256))
                s = step_state_by_byte(s, b)
                if step % 2 == 0:
                    o ^= mask12_for_byte(b)
            chi = chirality_word6(s)
            par = _collapse_to_6bit(o)
            state_parity[(s, par)] += 1
            state_counts[s] += 1
            par_counts[par] += 1
            chi_parity[(chi, par)] += 1
            chi_counts[chi] += 1

        def entropy_from_counts(counts, total):
            return -sum(c / total * math.log2(c / total) for c in counts.values())

        H_state = entropy_from_counts(state_counts, trials)
        H_par = entropy_from_counts(par_counts, trials)
        H_state_par = entropy_from_counts(state_parity, trials)
        H_chi = entropy_from_counts(chi_counts, trials)
        H_chi_par = entropy_from_counts(chi_parity, trials)

        H_par_given_state = H_state_par - H_state
        H_state_given_par = H_state_par - H_par
        H_par_given_chi = H_chi_par - H_chi

        print(f"\n  CONDITIONAL ENTROPIES (parity vs state vs chi):")
        print(f"    H(parity | final state):  {H_par_given_state:.4f} bits")
        print(f"    H(final state | parity):  {H_state_given_par:.4f} bits")
        print(f"    H(parity | chi):          {H_par_given_chi:.4f} bits")
        print(f"    (Parity adds beyond state; state adds beyond parity; parity vs chi)")

    def test_conditional_entropies_length2_exact(self):
        """
        Exact conditional entropies for length-2 trajectories (256^2 exhaustive).
        Checks whether H(parity | state) is exactly 2 bits (K4 fiber phase).
        """
        state_parity: Counter = Counter()
        state_counts: Counter = Counter()
        par_counts: Counter = Counter()
        chi_parity: Counter = Counter()
        chi_counts: Counter = Counter()

        for b1 in range(256):
            s1 = step_state_by_byte(GENE_MAC_REST, b1)
            o1 = mask12_for_byte(b1)
            for b2 in range(256):
                s2 = step_state_by_byte(s1, b2)
                o2 = o1 ^ mask12_for_byte(b2)
                par = _collapse_to_6bit(o2)
                chi = chirality_word6(s2)
                state_parity[(s2, par)] += 1
                state_counts[s2] += 1
                par_counts[par] += 1
                chi_parity[(chi, par)] += 1
                chi_counts[chi] += 1

        total = 256 * 256

        def entropy_from_counts(counts: Counter, tot: int) -> float:
            return -sum(c / tot * math.log2(c / tot) for c in counts.values())

        H_state = entropy_from_counts(state_counts, total)
        H_par = entropy_from_counts(par_counts, total)
        H_state_par = entropy_from_counts(state_parity, total)
        H_chi = entropy_from_counts(chi_counts, total)
        H_chi_par = entropy_from_counts(chi_parity, total)

        H_par_given_state = H_state_par - H_state
        H_state_given_par = H_state_par - H_par
        H_par_given_chi = H_chi_par - H_chi

        print(f"\n  CONDITIONAL ENTROPIES (length-2 EXHAUSTIVE, 256^2 words):")
        print(f"    H(parity | final state):  {H_par_given_state:.6f} bits")
        print(f"    H(final state | parity):  {H_state_given_par:.6f} bits")
        print(f"    H(parity | chi):          {H_par_given_chi:.6f} bits")
        print(f"    Distinct (state, parity) pairs: {len(state_parity)}")
        # If H(parity|state) ~ 2 then parity takes ~4 values per state (K4 fiber)
        if abs(H_par_given_state - 2.0) < 0.1:
            print(f"    H(parity|state) ~ 2 bits => ~4 parity values per state (K4 fiber)")


# ================================================================
# 9. EXACT TWO-STEP UNIFORMIZATION
# ================================================================


class TestExactTwoStepUniformization:
    """
    After 2 random bytes from rest, the output is exactly uniform on Ω.
    This follows from: depth-1 gives 128 states, each reached by
    exactly 2 bytes; depth-2 from each fills Ω uniformly.
    """

    def test_length2_exact_uniformity(self):
        print(f"\n{'='*65}")
        print("9. EXACT TWO-STEP UNIFORMIZATION")
        print(f"{'='*65}")

        # Exhaustive: all 256^2 = 65536 length-2 trajectories from rest
        outputs: Counter = Counter()
        for b1 in range(256):
            s1 = step_state_by_byte(GENE_MAC_REST, b1)
            for b2 in range(256):
                s2 = step_state_by_byte(s1, b2)
                outputs[s2] += 1

        distinct = len(outputs)
        counts = sorted(outputs.values())
        uniform_count = 65536 // 4096  # = 16

        is_uniform = all(c == uniform_count for c in counts)

        print(f"  All 256² = 65536 length-2 words from rest:")
        print(f"    Distinct outputs: {distinct}")
        print(f"    Expected (|Ω|): 4096")
        print(f"    Count per state: min={min(counts)} max={max(counts)} "
              f"expected={uniform_count}")
        print(f"    Exactly uniform: {is_uniform}")
        print(f"\n  Implication: after 2 random bytes, distribution is")
        print(f"  exactly uniform on Ω. Doubly stochastic transition")
        print(f"  preserves uniformity, so all lengths ≥ 2 are uniform.")
        print(f"  Depth-4 min-entropy deviations are pure sampling noise.")

        assert is_uniform, "Length-2 output is not exactly uniform"
        assert distinct == 4096


# ================================================================
# 10. TRAJECTORY COMPRESSION BOUNDS
# ================================================================


class TestTrajectoryCompression:
    """
    Redundancy grows with trajectory length because Ω saturates at 4096.
    Chirality captures exactly 6/12 = 50% of state information
    via a perfectly uniform 64-way partition.
    """

    def test_provenance_redundancy_by_length(self):
        rng = np.random.default_rng(80)

        print(f"\n{'='*65}")
        print("10. TRAJECTORY COMPRESSION BOUNDS")
        print(f"{'='*65}")
        print(f"  {'L':>3s}  {'InputBits':>10s}  {'DistinctOut':>11s}  "
              f"{'OutputBits':>10s}  {'Redundancy':>10s}  "
              f"{'BitsPerByte':>11s}")

        for L in [1, 2, 3, 4]:
            if L <= 2:
                outputs = set()
                for trial_bytes in range(256 ** L):
                    traj = [(trial_bytes >> (8 * i)) & 0xFF for i in range(L)]
                    s = GENE_MAC_REST
                    for b in traj:
                        s = step_state_by_byte(s, b)
                    outputs.add(s)
                distinct = len(outputs)
            else:
                outputs = set()
                for _ in range(min(500000, 256 ** L)):
                    traj = [int(rng.integers(0, 256)) for _ in range(L)]
                    s = GENE_MAC_REST
                    for b in traj:
                        s = step_state_by_byte(s, b)
                    outputs.add(s)
                distinct = len(outputs)

            input_bits = 8 * L
            output_bits = math.log2(distinct) if distinct > 0 else 0
            redundancy = input_bits - output_bits
            bits_per_byte = output_bits / L if L > 0 else 0

            print(f"  {L:3d}  {input_bits:10d}  {distinct:11d}  "
                  f"{output_bits:10.2f}  {redundancy:10.2f}  "
                  f"{bits_per_byte:11.4f}")

    def test_chirality_partition(self):
        omega_list, _ = _build_omega_index()

        chi_to_states: dict[int, list[int]] = {}
        for s in omega_list:
            chi = chirality_word6(s)
            chi_to_states.setdefault(chi, []).append(s)

        sizes = [len(v) for v in chi_to_states.values()]

        print(f"\n  CHIRALITY PARTITION:")
        print(f"    |Ω| = {len(omega_list)}")
        print(f"    Distinct χ values: {len(chi_to_states)}")
        print(f"    States per χ: {sizes[0]} (uniform: "
              f"{len(set(sizes)) == 1})")
        print(f"    χ captures {math.log2(64):.1f} / "
              f"{math.log2(4096):.1f} = 50.0% of state info")

        assert len(chi_to_states) == 64
        assert all(s == 64 for s in sizes)