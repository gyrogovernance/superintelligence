"""
Moments physics: tests that resolve open questions about the Router as a physical system.

The Router is a classical realization of exact stabilizer-class quantum dynamics on a paired
6-spin system. These tests are not restatements of existing physics tests; they test new claims:

1. Boundary vs volume normalization of physical capacity (holographic prediction).
2. 6-spin representation: isomorphism, transition law, magnetization, correlations.
3. Depth-4 frame as certification atom (strictly more informative than raw state).
4. CSM capacity with |Omega| = 4096.

Does not mix with legacy moments test files. Uses kernel API and physics test helpers only.
"""

from __future__ import annotations

import math

import numpy as np

from src.api import depth4_mask_projection48, mask12_for_byte
from src.constants import (
    GENE_MAC_REST,
    byte_to_intron,
    step_state_by_byte,
    unpack_state,
)

# -----
# Test 1: Boundary vs volume capacity and holographic prediction
# -----

F_CS = 9_192_631_770
OMEGA = 4096
HORIZON = 64


class TestBoundaryVsVolume:
    """
    Test: boundary vs volume normalization of the physical capacity.

    The Router has |H| = 64 (boundary) and |Omega| = 4096 (bulk) with |H|^2 = |Omega|.
    If the holography is physical, boundary- and volume-normalized capacities
    should be related by a Router-intrinsic quantity (|H|).
    """

    def test_physical_counts(self):
        """Compute N_vol and N_area and their ratio."""
        n_vol = (4 / 3) * math.pi * F_CS**3
        n_area = 4 * math.pi * F_CS**2

        ratio = n_vol / n_area
        assert abs(ratio - F_CS / 3) / (F_CS / 3) < 1e-10

        csm_vol = n_vol / OMEGA
        csm_area = n_area / HORIZON
        csm_ratio = csm_vol / csm_area

        predicted_ratio = F_CS / (3 * HORIZON)
        assert abs(csm_ratio - predicted_ratio) / predicted_ratio < 1e-10

        print("\n  N_vol:  %e" % n_vol)
        print("  N_area: %e" % n_area)
        print("  CSM_vol:  %e" % csm_vol)
        print("  CSM_area: %e" % csm_area)
        print("  Ratio CSM_vol/CSM_area: %e" % csm_ratio)
        print("  f_Cs / 192: %e" % predicted_ratio)
        print("  f_Cs / 192 = %d / 192 = %.2f" % (F_CS, F_CS / 192))

    def test_holographic_capacity_identity(self):
        """
        The holographic identity |H|^2 = |Omega| implies a specific
        relationship between boundary and bulk capacities.
        """
        n_vol = (4 / 3) * math.pi * F_CS**3
        n_area = 4 * math.pi * F_CS**2

        phys_ratio = n_vol / n_area
        assert abs(phys_ratio - F_CS / 3) < 1

        router_factor = HORIZON
        csm_vol = n_vol / OMEGA
        csm_area = n_area / HORIZON

        assert abs(csm_vol / csm_area - phys_ratio / router_factor) < 1

        print("\n  Physical V/A ratio: %.2f" % phys_ratio)
        print("  Router horizon: %d" % router_factor)
        print("  Per-state ratio: %.2f" % (phys_ratio / router_factor))


# -----
# Test 2: 6-spin isomorphism and physical consequences
# -----


def bits_to_spins(component12: int) -> np.ndarray:
    """Convert 12-bit component to 6-spin vector in {-1, +1}."""
    spins = np.zeros(6, dtype=int)
    for i in range(6):
        bit_lo = (component12 >> (2 * i)) & 1
        bit_hi = (component12 >> (2 * i + 1)) & 1
        spins[i] = bit_hi - bit_lo
    return spins


def spins_to_bits(spins: np.ndarray) -> int:
    """Convert 6-spin vector back to 12-bit component."""
    result = 0
    for i in range(6):
        if spins[i] == 1:
            result |= 1 << (2 * i + 1)
        else:
            result |= 1 << (2 * i)
    return result


def mask_to_flip_vector(mask12: int) -> np.ndarray:
    """Convert 12-bit mask to 6-element flip vector in {0, 1}."""
    flips = np.zeros(6, dtype=int)
    for i in range(6):
        if (mask12 >> (2 * i)) & 3:
            flips[i] = 1
    return flips


class TestSixSpinIsomorphism:
    """
    Test: exact 6-spin isomorphism on Omega.
    Transition law in spin coordinates and physical consequences
    (magnetization, correlations) that may be hidden in bit coordinates.
    """

    def test_roundtrip_on_all_omega(self):
        """bits_to_spins and spins_to_bits are exact inverses on Omega."""
        from tests.physics.test_physics_1 import _bfs_omega

        omega, _, _ = _bfs_omega()

        for state in omega:
            a, b = unpack_state(state)
            sa = bits_to_spins(a)
            sb = bits_to_spins(b)

            assert all(s in (-1, 1) for s in sa)
            assert all(s in (-1, 1) for s in sb)

            assert spins_to_bits(sa) == a
            assert spins_to_bits(sb) == b

    def test_transition_in_spin_coordinates(self):
        """
        Verify the spin-coordinate transition law against the bit-level kernel
        for all 256 bytes from rest.
        """
        a_rest, b_rest = unpack_state(GENE_MAC_REST)
        sb = bits_to_spins(b_rest)

        for byte in range(256):
            intron = byte_to_intron(byte)
            alpha = intron & 1
            beta = (intron >> 7) & 1
            m12 = mask12_for_byte(byte)
            flips = mask_to_flip_vector(m12)

            sign_a = (-1) ** alpha
            sign_b = (-1) ** beta
            m_diag = np.array([(-1) ** f for f in flips])

            sa_next_pred = sign_a * sb
            sb_next_pred = sign_b * (m_diag * bits_to_spins(a_rest))

            next_state = step_state_by_byte(GENE_MAC_REST, byte)
            a_next, b_next = unpack_state(next_state)
            sa_next_actual = bits_to_spins(a_next)
            sb_next_actual = bits_to_spins(b_next)

            assert np.array_equal(sa_next_pred, sa_next_actual), (
                "byte %#x: A spin mismatch" % byte
            )
            assert np.array_equal(sb_next_pred, sb_next_actual), (
                "byte %#x: B spin mismatch" % byte
            )

    def test_transition_on_random_omega_states(self):
        """Same verification on random states within Omega."""
        from tests.physics.test_physics_1 import _bfs_omega

        omega, _, _ = _bfs_omega()
        omega_list = sorted(omega)

        rng = np.random.default_rng(42)
        samples = rng.choice(len(omega_list), size=200, replace=False)

        for idx in samples:
            state = omega_list[idx]
            a, b = unpack_state(state)
            sa = bits_to_spins(a)
            sb = bits_to_spins(b)

            for byte in rng.integers(0, 256, size=10):
                byte = int(byte)
                intron = byte_to_intron(byte)
                alpha = intron & 1
                beta = (intron >> 7) & 1
                m12 = mask12_for_byte(byte)
                flips = mask_to_flip_vector(m12)

                sign_a = (-1) ** alpha
                sign_b = (-1) ** beta
                m_diag = np.array([(-1) ** f for f in flips])

                sa_next_pred = sign_a * sb
                sb_next_pred = sign_b * (m_diag * sa)

                next_state = step_state_by_byte(state, byte)
                a_next, b_next = unpack_state(next_state)

                assert np.array_equal(sa_next_pred, bits_to_spins(a_next))
                assert np.array_equal(sb_next_pred, bits_to_spins(b_next))

    def test_magnetization_conservation(self):
        """
        In spin coordinates, total magnetization M = sum(a) + sum(b).
        Probe whether this has structure on Omega.
        """
        from tests.physics.test_physics_1 import _bfs_omega

        omega, _, _ = _bfs_omega()

        magnetizations = set()
        for state in omega:
            a, b = unpack_state(state)
            sa = bits_to_spins(a)
            sb = bits_to_spins(b)
            m_total = int(np.sum(sa) + np.sum(sb))
            magnetizations.add(m_total)

        print("\n  Magnetization values on Omega: %s" % sorted(magnetizations))
        print("  Number of distinct values: %d" % len(magnetizations))

        a_rest, b_rest = unpack_state(GENE_MAC_REST)
        m_rest = int(
            np.sum(bits_to_spins(a_rest)) + np.sum(bits_to_spins(b_rest))
        )
        print("  Rest state magnetization: %d" % m_rest)

        m_after = set()
        for byte in range(256):
            next_state = step_state_by_byte(GENE_MAC_REST, byte)
            a, b = unpack_state(next_state)
            m = int(np.sum(bits_to_spins(a)) + np.sum(bits_to_spins(b)))
            m_after.add(m)

        print("  Magnetizations reachable from rest in 1 step: %s" % sorted(m_after))

    def test_spin_correlation_matrix(self):
        """
        Spin-spin correlation matrix C_ij = <s_i * s_j> averaged over Omega.
        """
        from tests.physics.test_physics_1 import _bfs_omega

        omega, _, _ = _bfs_omega()

        corr = np.zeros((12, 12), dtype=float)
        for state in omega:
            a, b = unpack_state(state)
            s = np.concatenate([bits_to_spins(a), bits_to_spins(b)])
            corr += np.outer(s, s)

        corr /= len(omega)

        for i in range(12):
            assert abs(corr[i, i] - 1.0) < 1e-10

        print("\n  12-spin correlation matrix (averaged over Omega):")
        print("  A-A block (upper-left 6x6):")
        print(np.array2string(corr[:6, :6], precision=4, suppress_small=True))
        print("  B-B block (lower-right 6x6):")
        print(np.array2string(corr[6:, 6:], precision=4, suppress_small=True))
        print("  A-B block (upper-right 6x6):")
        print(np.array2string(corr[:6, 6:], precision=4, suppress_small=True))


# -----
# Test 3: Depth-4 frame as certification atom
# -----


def _frame_record(b0: int, b1: int, b2: int, b3: int) -> tuple[int, int, int]:
    """Compute the depth-4 frame record: (mask48, phi_a, phi_b)."""
    mask48 = depth4_mask_projection48(b0, b1, b2, b3)

    fams = []
    for b in (b0, b1, b2, b3):
        intron = byte_to_intron(b)
        a_bit = intron & 1
        b_bit = (intron >> 7) & 1
        fams.append((a_bit, b_bit))

    phi_a = fams[0][1] ^ fams[1][0] ^ fams[2][1] ^ fams[3][0]
    phi_b = fams[0][0] ^ fams[1][1] ^ fams[2][0] ^ fams[3][1]

    return (mask48, phi_a, phi_b)


def _apply_word(state: int, word: list[int]) -> int:
    """Apply a sequence of bytes and return final state."""
    s = state
    for b in word:
        s = step_state_by_byte(s, b)
    return s


class TestFrameAsCertificationAtom:
    """
    Test: depth-4 frame provides strictly better divergence detection
    than individual bytes. Frame record (mask48, phi_a, phi_b) vs raw state.
    """

    def test_frame_record_is_deterministic(self):
        """Same 4 bytes always produce the same frame record."""
        rng = np.random.default_rng(0)
        for _ in range(1000):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            r1 = _frame_record(*word)
            r2 = _frame_record(*word)
            assert r1 == r2

    def test_different_words_different_frame_records(self):
        """Words that differ only in family may share mask48 but differ in phi."""
        from tests.physics.test_physics_5 import _byte_from_micro_family

        micros = [1, 7, 13, 29]

        records = set()
        for f0 in range(4):
            for f1 in range(4):
                for f2 in range(4):
                    for f3 in range(4):
                        word = [
                            _byte_from_micro_family(micros[0], f0),
                            _byte_from_micro_family(micros[1], f1),
                            _byte_from_micro_family(micros[2], f2),
                            _byte_from_micro_family(micros[3], f3),
                        ]
                        r = _frame_record(*word)
                        records.add(r)

        mask48_values = {r[0] for r in records}
        assert len(mask48_values) == 1
        assert len(records) == 4

    def test_frame_detects_single_byte_change(self):
        """Changing any single byte in a 4-byte frame always changes the frame record."""
        rng = np.random.default_rng(42)
        detected = 0
        total = 0

        for _ in range(1000):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            r_orig = _frame_record(*word)

            for pos in range(4):
                alt_byte = (word[pos] + 1) % 256
                word_alt = word.copy()
                word_alt[pos] = alt_byte
                r_alt = _frame_record(*word_alt)
                total += 1
                if r_alt != r_orig:
                    detected += 1

        print("\n  Single-byte changes detected by frame record: %d/%d" % (detected, total))
        assert detected == total

    def test_state_collision_with_different_frames(self):
        """
        Find pairs of 4-byte words that produce the same final state
        from rest but have different frame records.
        """
        rng = np.random.default_rng(123)
        state_to_frames: dict[int, set] = {}

        for _ in range(50000):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            final = _apply_word(GENE_MAC_REST, word)
            fr = _frame_record(*word)
            state_to_frames.setdefault(final, set()).add(fr)

        multi = {s: frs for s, frs in state_to_frames.items() if len(frs) > 1}

        print("\n  Distinct final states observed: %d" % len(state_to_frames))
        print("  States with multiple frame records: %d" % len(multi))
        if multi:
            example = next(iter(multi.items()))
            print("  Example state %#08x: %d distinct frames" % (example[0], len(example[1])))

        assert len(multi) > 0, (
            "frame records must distinguish histories that collapse to same state"
        )

    def test_frame_sequence_divergence_localizes_to_frame(self):
        """Two logs that diverge at byte k have identical frame sequences up to frame (k//4 - 1)."""
        rng = np.random.default_rng(99)

        for _ in range(100):
            log = [int(rng.integers(0, 256)) for _ in range(20)]

            flip_pos = int(rng.integers(0, 20))
            log_alt = log.copy()
            log_alt[flip_pos] = (log_alt[flip_pos] + 1) % 256

            frames_orig = []
            frames_alt = []
            for i in range(0, 20, 4):
                frames_orig.append(_frame_record(*log[i : i + 4]))
                frames_alt.append(_frame_record(*log_alt[i : i + 4]))

            affected_frame = flip_pos // 4
            for i in range(affected_frame):
                assert frames_orig[i] == frames_alt[i], (
                    "Frame %d should match (flip at byte %d, frame %d)"
                    % (i, flip_pos, affected_frame)
                )

            assert frames_orig[affected_frame] != frames_alt[affected_frame], (
                "Frame %d should differ (flip at byte %d)" % (affected_frame, flip_pos)
            )


# -----
# Test 4: CSM capacity with new Omega
# -----

OMEGA_NEW = 4096
HORIZON_NEW = 64
POPULATION = 8_100_000_000
UHI_PER_YEAR = 87_600


class TestCSMWithNewOmega:
    """CSM capacity with |Omega| = 4096. Boundary-normalized alternative for comparison."""

    def test_n_phys_unchanged(self):
        """N_phys depends only on f_Cs, not on the Router."""
        n_phys = (4 / 3) * math.pi * F_CS**3
        assert abs(n_phys - 3.253930e30) / 3.253930e30 < 1e-4

    def test_c_cancellation_unchanged(self):
        """Speed of light still cancels."""
        c1 = 299_792_458
        c2 = 2 * c1
        c3 = 0.1 * c1

        def n_phys_with_c(c: float) -> float:
            v = (4 / 3) * math.pi * (c * 1) ** 3
            lam = c / F_CS
            return v / lam**3

        n1 = n_phys_with_c(c1)
        n2 = n_phys_with_c(c2)
        n3 = n_phys_with_c(c3)

        assert abs(n1 - n2) / n1 < 1e-10
        assert abs(n1 - n3) / n1 < 1e-10

    def test_csm_new(self):
        """CSM = N_phys / 4096."""
        n_phys = (4 / 3) * math.pi * F_CS**3
        csm = n_phys / OMEGA_NEW

        csm_old = n_phys / 65536
        assert abs(csm / csm_old - 16.0) < 1e-10

        global_demand = POPULATION * UHI_PER_YEAR
        coverage_years = csm / global_demand

        print("\n  N_phys: %e" % n_phys)
        print("  |Omega|: %d" % OMEGA_NEW)
        print("  CSM: %e MU" % csm)
        print("  CSM (old): %e MU" % csm_old)
        print("  Ratio new/old: %.1fx" % (csm / csm_old))
        print("  Global UHI demand/year: %e MU" % global_demand)
        print("  Coverage: %.2e years" % coverage_years)

        assert coverage_years > 1e11

    def test_identity_scaling_with_new_horizon(self):
        """Identities = |H| * 128^n for path length n."""
        for n in range(1, 6):
            identities = HORIZON_NEW * (128**n)
            sufficient = identities >= 10_000_000_000
            print("  n=%d: %20d identities %s" % (n, identities, "[OK]" if sufficient else "[--]"))

        assert HORIZON_NEW * 128**4 > 10_000_000_000
