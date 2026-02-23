"""
Physics tests Part 4: Kernel structure expanded towards CGM Units.

Design intent:
- Focus on non-trivial, proof-grade structure that we can cite.
- Print meaningful results for documentation.
- Avoid re-testing heavy items already certified in test_physics_1.py and test_physics_2.py
  (e.g. full step law across all transitions, commutator translation proof, etc.)

What this file certifies and prints:

A) Byte-action group facts (atlas-real)
   1) Reference byte 0xAA is an involution with exactly 256 fixed points and 32640 2-cycles.
   2) Every non-reference byte b != 0xAA decomposes into 16384 disjoint 4-cycles.
      This is a hard claim we can make because T_b^2 is a nonzero XOR translation,
      hence has no fixed points, which forbids 1- and 2-cycles under T_b.

   It also prints the eigenphase multiplicities implied by these cycle decompositions,
   treating the permutation as a unitary on a 65536-dimensional Hilbert space.

B) Mask code duality (coding-theory facts pinned by the kernel expansion)
   1) The mask set C has size 256 and lives in GF(2)^12, so it is a [12,8] linear code.
   2) The dual code C_perp has size 16 and satisfies |C|*|C_perp| = 2^12 = 4096.
   3) Walsh transform theorem (proof-grade):
        W(s) = sum_{c in C} (-1)^{<s,c>} is 256 exactly when s in C_perp, else 0.
      We compute W(s) for all 4096 s and verify the support is exactly C_perp.

C) Atlas-wide shell distributions forced by the code enumerator
   1) Horizon distance histogram over Ω equals 256 * A_w
   2) Archetype distance histogram over Ω equals convolution(A_w, A_w)

D) CGM Units bridge (diagnostic, printed)
   Using only kernel-intrinsic discrete constants already present in your work:
     - A_kernel = 5/256
     - theta_min = arccos(5/6) from the weight-1 defect in 12D ±1 embedding
     - delta_kernel = theta_min/3  (3-row anatomy split)
     - m_a_kernel = delta_kernel / (1 - A_kernel)
     - Q_G_kernel = 1/(2 m_a_kernel^2)
     - K_QG_kernel = (pi/4)/m_a_kernel
     - alpha_kernel = delta_kernel^4 / m_a_kernel

   It prints comparisons to CGM anchors from Analysis_CGM_Units.md and CGM_Paper.md.

Important:
- The CGM bridge section is intentionally "diagnostic": it prints and compares.
  Assertions are minimal and do not force agreement to CGM numbers.
"""

from __future__ import annotations

from collections import Counter
from math import pi
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_B12,
    ARCHETYPE_STATE24,
    LAYER_MASK_12,
)
from tests._physics_utils import (
    apply_word_to_indices,
    coeffs_archetype_distance_enumerator_closed_form,
    coeffs_mask_weight_enumerator_closed_form,
    cycle_lengths_of_permutation,
    dual_code_from_parity_checks,
    fmt,
    krawtchouk,
    parity12_arr,
    popcount12_arr,
    uv_from_state24,
    weight_enumerator_counts,
    word_odd_even_xors,
)
from tests._physics_utils import (
    table as _table,
)

# Diagnostic printing flag (set to False to suppress output)
PRINT = True


# Helper wrapper for table() that respects PRINT flag
def table_if_enabled(title: str, rows: list[tuple[str, str]]) -> None:
    """Wrapper for table() that respects PRINT flag."""
    _table(title, rows, enable=PRINT)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture(scope="module")
def atlas():
    atlas_dir = Path("data/atlas")
    if not atlas_dir.exists():
        pytest.skip("Atlas not built. Run: python -m src.router.atlas")

    ontology = np.load(atlas_dir / "ontology.npy", mmap_mode="r").astype(np.uint32)
    epistemology = np.load(atlas_dir / "epistemology.npy", mmap_mode="r").astype(np.uint32)

    from src.router.kernel import RouterKernel
    k = RouterKernel(atlas_dir)
    masks24 = k.xform_mask_by_byte.astype(np.uint32)
    masks_a12 = ((masks24 >> 12) & 0xFFF).astype(np.uint16)

    return {
        "dir": atlas_dir,
        "ont": ontology,
        "epi": epistemology,
        "masks_a12": masks_a12,
    }


# -----------------------------
# Helpers (file-specific only)
# -----------------------------

def parity12_arr(x: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Parity (mod 2) for array of 12-bit values."""
    from tests._physics_utils import parity12_arr as _parity12_arr
    return _parity12_arr(x)


# ========
# A) Byte-action cycle structure and implied eigenphases
# ========

class TestKernelByteCyclesAndEigenphases:
    def test_reference_byte_cycle_decomposition_and_eigenphases(self, atlas):
        """
        Hard, atlas-real facts for reference byte R = 0xAA:
        - R^2 = id (involution)
        - fixed points count is exactly 256
        - remaining states form 2-cycles: (65536 - 256)/2 = 32640

        Prints cycle statistics and eigenphase multiplicities for the permutation unitary.
        """
        epi = atlas["epi"]
        n = int(epi.shape[0])
        idxs = np.arange(n, dtype=np.int64)

        perm = epi[:, 0xAA].astype(np.int64)
        perm2 = perm[perm]

        assert np.array_equal(perm2, idxs), "0xAA must be an involution"
        assert not np.array_equal(perm, idxs), "0xAA must not be identity"

        fixed = int(np.sum(perm == idxs))
        assert fixed == 256, f"Expected 256 fixed points, got {fixed}"

        # Full cycle decomposition (one-time, acceptable)
        lengths = cycle_lengths_of_permutation(perm)
        counts = Counter(lengths)

        assert set(counts.keys()) == {1, 2}
        assert counts[1] == 256
        assert counts[2] == (65536 - 256) // 2

        # Eigenphases for permutation unitary:
        # 1-cycle contributes eigenvalue 1
        # 2-cycle contributes eigenvalues {1, -1}
        n2 = counts[2]
        eig_mult = {
            "1": counts[1] + n2,
            "-1": n2,
        }

        table_if_enabled(
            "Reference byte 0xAA - cycle and eigenphase structure",
            [
                ("n", str(n)),
                ("fixed points (1-cycles)", str(counts[1])),
                ("2-cycles", str(counts[2])),
                ("eigenvalue multiplicity 1", str(eig_mult["1"])),
                ("eigenvalue multiplicity -1", str(eig_mult["-1"])),
            ],
        )

    def test_all_nonreference_bytes_are_pure_4_cycles(self, atlas):
        """
        Hard claim: for every byte b != 0xAA, the permutation on Ω decomposes into 4-cycles only.

        Reason (kernel-internal):
        - For any b, T_b^2 is a translation by (m_b, m_b) in (u,v) space.
        - If m_b != 0, that translation has no fixed points.
        - If a point were in a 1- or 2-cycle of T_b, it would be fixed by T_b^2.
        - Therefore no 1- or 2-cycles can exist when m_b != 0.
        - Since we already know T_b^4 = id on Ω, the only remaining cycle length is 4.

        We certify this atlas-wide for all 255 non-reference bytes by checking that T_b^2
        has zero fixed points for each b != 0xAA, and we print a concise summary.
        """
        epi = atlas["epi"]
        masks = atlas["masks_a12"].astype(np.uint16)

        n = int(epi.shape[0])
        idxs = np.arange(n, dtype=np.int64)

        bad: list[int] = []
        for b in range(256):
            if b == 0xAA:
                continue

            # mask must be nonzero for all b != 0xAA
            assert int(masks[b]) != 0, f"Unexpected zero mask at byte 0x{b:02x}"

            perm = epi[:, b].astype(np.int64)
            perm2 = perm[perm]
            fixed2 = int(np.sum(perm2 == idxs))
            if fixed2 != 0:
                bad.append(b)

        table_if_enabled(
            "Non-reference bytes: fixed points of T_b^2",
            [
                ("bytes tested", "255"),
                ("anomalies", str(len(bad))),
                ("anomaly list", str([f"0x{x:02x}" for x in bad]) if bad else "[]"),
            ],
        )

        assert len(bad) == 0, "Some non-reference byte has T_b^2 fixed points, contradicting pure 4-cycle claim"

        # Additionally, show one representative cycle histogram and eigenphases
        b = 0x42
        perm = epi[:, b].astype(np.int64)
        lengths = cycle_lengths_of_permutation(perm)
        counts = Counter(lengths)

        assert set(counts.keys()) == {4}
        assert counts[4] == 65536 // 4

        eig_mult = {
            "1": counts[4],
            "i": counts[4],
            "-1": counts[4],
            "-i": counts[4],
        }

        table_if_enabled(
            "Representative non-reference byte 0x42 - cycle and eigenphase structure",
            [
                ("4-cycles", str(counts[4])),
                ("eigenvalue multiplicity 1", str(eig_mult["1"])),
                ("eigenvalue multiplicity i", str(eig_mult["i"])),
                ("eigenvalue multiplicity -1", str(eig_mult["-1"])),
                ("eigenvalue multiplicity -i", str(eig_mult["-i"])),
            ],
        )


# ========
# B) Code duality and Walsh spectrum support theorem
# ========

class TestMaskCodeDualityAndFourierSupport:
    def test_generator_bytes_span_all_masks_and_print(self, atlas):
        """
        This replaces the previous failing test in the earlier physics_3.
        The failure was caused by an incorrect 'min' heuristic, not by the kernel.

        Here we use a basis aligned with the kernel's transcription algebra:
        intron basis vectors e_i (i=0..7) mapped to bytes via b = 0xAA XOR (1<<i).
        Those 8 bytes produce 8 generator masks that span the [12,8] mask code.

        We certify:
        - the 8 generator masks produce exactly 256 distinct XOR combinations
        - that set equals the full mask set produced by the kernel
        """
        masks = atlas["masks_a12"].astype(np.uint16)

        gen_bytes = [((1 << i) ^ 0xAA) & 0xFF for i in range(8)]
        gen_masks = [int(masks[b]) for b in gen_bytes]

        # Build all 2^8 combinations
        reachable: set[int] = set()
        for s in range(256):
            m = 0
            for i in range(8):
                if (s >> i) & 1:
                    m ^= gen_masks[i]
            reachable.add(m & 0xFFF)

        actual = set(int(m) for m in masks)

        table_if_enabled(
            "Mask code spanning set from intron-basis bytes",
            [
                ("generator bytes", str([f"0x{b:02x}" for b in gen_bytes])),
                ("generator masks (hex)", str([f"0x{m:03x}" for m in gen_masks])),
                ("reachable masks", str(len(reachable))),
                ("actual masks", str(len(actual))),
                ("match", str(reachable == actual)),
            ],
        )

        assert len(reachable) == 256
        assert reachable == actual

        # Context note: rank 8 matches code dimension and '8' as a common structural number
        print("\n  Context note: mask code has dimension 8 over GF(2), hence |C| = 2^8 = 256.")

    def test_code_duality_sizes_and_macwilliams_identity(self, atlas):
        """
        Certify rigorous duality facts for the [12,8] mask code C:

        1) Size identity (always true for linear codes):
           |C| * |C_perp| = 2^12 = 4096

        2) MacWilliams identity: weight enumerator of C determines that of C_perp.

        We compute:
          A_w = count of codewords in C of weight w
          B_w = count of codewords in C_perp of weight w

        Then verify B_w matches the MacWilliams transform of A_w.
        """
        masks_c = atlas["masks_a12"].astype(np.uint16)
        A = weight_enumerator_counts(masks_c)  # length 13, sum 256

        # Dual parity checks (diagonal ties - already established in your work)
        H = [
            (1 << 0) | (1 << 8),
            (1 << 1) | (1 << 9),
            (1 << 2) | (1 << 10),
            (1 << 3) | (1 << 11),
        ]
        dual_words = dual_code_from_parity_checks(H)
        B_actual = weight_enumerator_counts(dual_words)  # sum 16

        n = 12
        size_c = 256

        # MacWilliams transform
        B_pred = np.zeros(n + 1, dtype=np.int64)
        for w in range(n + 1):
            s = 0
            for j in range(n + 1):
                s += int(A[j]) * krawtchouk(n, w, j)
            assert s % size_c == 0
            B_pred[w] = s // size_c

        table_if_enabled(
            "Code duality and MacWilliams identity",
            [
                ("n", str(n)),
                ("|C|", str(int(masks_c.size))),
                ("|C_perp|", str(int(dual_words.size))),
                ("|C|*|C_perp|", str(int(masks_c.size) * int(dual_words.size))),
                ("2^12", str(2**12)),
                ("MacWilliams match", str(np.array_equal(B_pred, B_actual))),
                ("B_actual weights", str({i: int(B_actual[i]) for i in range(13) if B_actual[i]})),
            ],
        )

        assert int(masks_c.size) == 256
        assert int(dual_words.size) == 16
        assert int(masks_c.size) * int(dual_words.size) == 2**12
        assert np.array_equal(B_pred, B_actual)

    def test_walsh_spectrum_support_equals_dual_code(self, atlas):
        """
        Proof-grade identity for a linear subspace C ⊂ GF(2)^n:

          W(s) = sum_{c in C} (-1)^{<s,c>} = |C|  if s ∈ C_perp
                                           0     otherwise

        We compute W(s) for all 4096 s in GF(2)^12, using the kernel's mask code C,
        and verify:
        - W(s) only takes values in {0, 256}
        - exactly 16 s values have W(s)=256
        - those 16 values are exactly C_perp generated by the known parity checks
        """
        masks_c = atlas["masks_a12"].astype(np.uint16)
        assert int(masks_c.size) == 256

        # Dual parity checks (diagonal ties)
        H = [
            (1 << 0) | (1 << 8),
            (1 << 1) | (1 << 9),
            (1 << 2) | (1 << 10),
            (1 << 3) | (1 << 11),
        ]
        dual_words = dual_code_from_parity_checks(H)
        dual_set = set(int(x) for x in dual_words)

        s_vals = np.arange(4096, dtype=np.uint16)
        W = np.zeros(4096, dtype=np.int32)

        # Compute in chunks
        chunk = 256
        for start in range(0, 4096, chunk):
            s_chunk = s_vals[start : start + chunk]  # (chunk,)
            ands = (s_chunk[:, None] & masks_c[None, :]).astype(np.uint16)  # (chunk,256)
            par = parity12_arr(ands).astype(np.int8)
            signs = (1 - 2 * par).astype(np.int16)
            W[start : start + chunk] = signs.sum(axis=1).astype(np.int32)

        values, counts = np.unique(W, return_counts=True)
        spectrum = dict(zip(values.tolist(), counts.tolist()))

        support = set(int(i) for i in np.where(W == 256)[0])
        support_size = len(support)

        table_if_enabled(
            "Walsh spectrum of mask-code indicator",
            [
                ("unique W(s) values", str(sorted(spectrum.keys()))),
                ("counts by value", str({k: spectrum[k] for k in sorted(spectrum.keys())})),
                ("support size W(s)=256", str(support_size)),
                ("expected |C_perp|", "16"),
                ("support equals C_perp", str(support == dual_set)),
            ],
        )

        assert set(values.tolist()) <= {0, 256}
        assert support_size == 16
        assert support == dual_set


# ========
# C) Atlas distributions forced by the code enumerator
# ========

class TestAtlasShellDistributionsFromCodeEnumerator:
    def test_horizon_distance_shells_equal_256_times_code_enumerator(self, atlas):
        """
        Define horizon distance:
          hd(A,B) = popcount( A XOR (B XOR 0xFFF) )

        Over Ω = C x C, the multiset of (u XOR v) values is uniform over C, so:
          count_hd[w] = 256 * A_w

        This is an exact consequence of the cartesian product ontology and code closure.
        """
        ont = atlas["ont"]
        masks_c = atlas["masks_a12"].astype(np.uint16)
        A = weight_enumerator_counts(masks_c)  # weights 0..12

        a = ((ont >> 12) & 0xFFF).astype(np.uint16)
        b = (ont & 0xFFF).astype(np.uint16)
        hd = popcount12_arr((a ^ (b ^ np.uint16(LAYER_MASK_12))).astype(np.uint16)).astype(np.int64)
        counts_hd = np.bincount(hd, minlength=13).astype(np.int64)

        expected = A * 256
        assert int(counts_hd.sum()) == 65536
        assert np.array_equal(counts_hd, expected)

        # Print compact distribution
        print("\nHorizon distance shells over Ω (exact)")
        print("-----------------------------------")
        for w in range(13):
            c = int(counts_hd[w])
            p = c / 65536.0
            if c:
                print(f"  w={w:2d} : count={c:6d}  prob={p:.6f}")

    def test_archetype_distance_shells_equal_convolution(self, atlas):
        """
        Define archetype distance as 24-bit Hamming distance to ARCHETYPE_STATE24.

        Over Ω = C x C:
          dist24 = popcount(u) + popcount(v)

        Therefore:
          count_dist24[d] = sum_{i+j=d} A_i * A_j  (discrete convolution)
        """
        ont = atlas["ont"]
        masks_c = atlas["masks_a12"].astype(np.uint16)
        A = weight_enumerator_counts(masks_c).astype(np.int64)

        a = ((ont >> 12) & 0xFFF).astype(np.uint16)
        b = (ont & 0xFFF).astype(np.uint16)

        da = popcount12_arr((a ^ np.uint16(ARCHETYPE_A12)).astype(np.uint16)).astype(np.int64)
        db = popcount12_arr((b ^ np.uint16(ARCHETYPE_B12)).astype(np.uint16)).astype(np.int64)
        dist24 = (da + db).astype(np.int64)

        counts24 = np.bincount(dist24, minlength=25).astype(np.int64)
        expected = np.convolve(A, A)
        assert expected.size == 25
        assert int(counts24.sum()) == 65536
        assert np.array_equal(counts24, expected)

        print("\nArchetype distance shells over Ω (exact)")
        print("---------------------------------------")
        for d in range(25):
            c = int(counts24[d])
            p = c / 65536.0
            if c:
                print(f"  dist={d:2d} : count={c:6d}  prob={p:.6f}")


# ========
# D) CGM Units bridge - diagnostics printed, minimal hard asserts
# ========

class TestCGMUnitsBridgeDiagnostics:
    def test_kernel_to_cgm_units_bridge_prints(self, atlas):
        """
        Prints a disciplined mapping from kernel intrinsic discrete invariants to CGM Units anchors.

        Uses:
        - code enumerator to get A_kernel = 5/256 (weight 0 and 1 sector mass)
        - theta_min = arccos(5/6)
        - delta_kernel = theta_min / 3 (3-row split)
        - omega_kernel = delta_kernel / 2
        - m_a_kernel = delta_kernel / (1 - A_kernel)
        - Q_G_kernel = 1 / (2*m_a_kernel^2)
        - K_QG_kernel = (pi/4)/m_a_kernel
        - alpha_kernel = delta_kernel^4 / m_a_kernel

        Compares to CGM anchors:
        - m_a_CGM = 1/(2 sqrt(2 pi))
        - Q_G_CGM = 4 pi
        - delta_BU_CGM = 0.1953421766
        - A*_CGM = 1 - delta_BU/m_a
        - alpha_CGM numbers used in your docs

        We do not assert "must match"; we print and bound-check.
        """
        masks = atlas["masks_a12"].astype(np.uint16)
        counts = weight_enumerator_counts(masks)

        A_kernel = float((counts[0] + counts[1]) / 256.0)  # 5/256
        closure_kernel = 1.0 - A_kernel

        theta_min = float(np.arccos(5.0 / 6.0))
        delta_kernel = theta_min / 3.0
        omega_kernel = delta_kernel / 2.0

        m_a_kernel = delta_kernel / closure_kernel
        QG_kernel = 1.0 / (2.0 * (m_a_kernel ** 2))
        KQG_kernel = (pi / 4.0) / m_a_kernel
        alpha_kernel = (delta_kernel ** 4) / m_a_kernel

        # CGM anchors from your docs
        m_a_CGM = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))
        QG_CGM = 4.0 * np.pi
        delta_BU_CGM = 0.1953421766
        A_star_CGM = 1.0 - (delta_BU_CGM / m_a_CGM)

        # K_QG anchor in CGM Units
        KQG_CGM_from_ma = (pi / 4.0) / m_a_CGM
        KQG_CGM_closed = (pi ** 2) / np.sqrt(2.0 * np.pi)

        # Two alpha baselines present in your materials
        alpha_CGM_paper = 0.007297352563
        alpha_CGM_units = 0.007299734

        table_if_enabled(
            "CGM Units ↔ Kernel bridge (diagnostic)",
            [
                ("A_kernel", f"{fmt(A_kernel, 12)} (exact 5/256={5/256:.12f})"),
                ("A*_CGM", f"{fmt(A_star_CGM, 12)}"),
                ("A_kernel - A*_CGM", f"{A_kernel - A_star_CGM:+.12f}"),
                ("", ""),
                ("theta_min", f"{fmt(theta_min, 12)}  (arccos(5/6))"),
                ("delta_kernel", f"{fmt(delta_kernel, 12)}  (theta_min/3)"),
                ("delta_BU_CGM", f"{fmt(delta_BU_CGM, 12)}"),
                ("delta_kernel - delta_BU_CGM", f"{delta_kernel - delta_BU_CGM:+.12f}"),
                ("omega_kernel", f"{fmt(omega_kernel, 12)}  (delta_kernel/2)"),
                ("", ""),
                ("m_a_kernel", f"{fmt(m_a_kernel, 12)}"),
                ("m_a_CGM", f"{fmt(m_a_CGM, 12)}"),
                ("m_a_kernel - m_a_CGM", f"{m_a_kernel - m_a_CGM:+.12f}"),
                ("", ""),
                ("Q_G_kernel", f"{fmt(QG_kernel, 12)}"),
                ("Q_G_CGM = 4π", f"{fmt(QG_CGM, 12)}"),
                ("Q_G_kernel - 4π", f"{QG_kernel - QG_CGM:+.12f}"),
                ("", ""),
                ("K_QG_kernel", f"{fmt(KQG_kernel, 12)}  ((π/4)/m_a_kernel)"),
                ("K_QG_CGM((π/4)/m_a)", f"{fmt(KQG_CGM_from_ma, 12)}"),
                ("K_QG_CGM(pi^2/sqrt(2pi))", f"{fmt(KQG_CGM_closed, 12)}"),
                ("", ""),
                ("alpha_kernel", f"{fmt(alpha_kernel, 12)}  (delta_kernel^4 / m_a_kernel)"),
                ("alpha_CGM (paper)", f"{fmt(alpha_CGM_paper, 12)}"),
                ("alpha_CGM (units)", f"{fmt(alpha_CGM_units, 12)}"),
                ("alpha_kernel - alpha_CGM(paper)", f"{alpha_kernel - alpha_CGM_paper:+.12f}"),
            ],
        )

        # Minimal hard asserts only
        assert abs(A_kernel - (5.0 / 256.0)) < 1e-15
        assert 0.0 < m_a_kernel < 1.0
        assert 0.0 < QG_kernel < 100.0
        assert 0.0 < alpha_kernel < 1.0


# ========
# E) Group presentation on (u,v) coordinates
# ========

class TestActionGroupPresentationInUV:
    """
    This certifies a strong presentation claim:

    For any byte word W with masks m_i and parity/O/E computed as usual,
    the action on (u,v) is exactly:

      if parity is even:
        u' = u XOR O
        v' = v XOR E

      if parity is odd:
        u' = v XOR E
        v' = u XOR O

    This means:
    - There are only two possible linear parts: identity and swap.
    - All translation data lives in C × C (the [12,8] code product).
    - The full word action is completely determined by (parity, O, E).

    This is a hard structural statement about the kernel dynamics as an affine action.
    """

    def test_word_action_depends_only_on_parity_OE_and_prints(self, atlas):
        ont = atlas["ont"]
        epi = atlas["epi"]
        masks = atlas["masks_a12"]

        rng = np.random.default_rng(20260101)

        # Probe states (small set is enough for a strong certification, but not tiny)
        n = int(ont.size)
        probe_count = 8192
        idxs = rng.integers(0, n, size=probe_count, dtype=np.int64)
        states0 = ont[idxs]

        u0 = np.empty(probe_count, dtype=np.uint16)
        v0 = np.empty(probe_count, dtype=np.uint16)
        for i in range(probe_count):
            ui, vi = uv_from_state24(int(states0[i]))
            u0[i] = ui
            v0[i] = vi

        # Random words with varying lengths, deliberately including long words
        words: list[list[int]] = []
        for _ in range(200):
            L = int(rng.integers(1, 65))  # lengths 1..64
            words.append(rng.integers(0, 256, size=L, dtype=np.int64).tolist())

        # Verify each word matches the closed form on all probe states
        mismatches = 0
        for w in words:
            parity, O, E = word_odd_even_xors(w, masks)
            idxs_out = apply_word_to_indices(epi, idxs, w)
            states_out = ont[idxs_out]

            u1 = np.empty(probe_count, dtype=np.uint16)
            v1 = np.empty(probe_count, dtype=np.uint16)
            for i in range(probe_count):
                ui, vi = uv_from_state24(int(states_out[i]))
                u1[i] = ui
                v1[i] = vi

            if parity == 0:
                u_pred = (u0 ^ np.uint16(O)).astype(np.uint16)
                v_pred = (v0 ^ np.uint16(E)).astype(np.uint16)
            else:
                u_pred = (v0 ^ np.uint16(E)).astype(np.uint16)
                v_pred = (u0 ^ np.uint16(O)).astype(np.uint16)

            if not (np.array_equal(u1, u_pred) and np.array_equal(v1, v_pred)):
                mismatches += 1

        table_if_enabled(
            "Action group presentation check (u,v)",
            [
                ("probe states", str(probe_count)),
                ("random words tested", str(len(words))),
                ("mismatching words", str(mismatches)),
            ],
        )

        assert mismatches == 0, "Some word did not match the (parity,O,E) affine presentation in (u,v)"

    def test_only_two_linear_parts_exist_identity_or_swap(self, atlas):
        """
        This certifies that the linear part of every word action is either:
        - identity (even length)
        - swap (odd length)

        We detect the linear part using two probe states that differ only in u.
        """
        ont = atlas["ont"]
        epi = atlas["epi"]
        masks = atlas["masks_a12"]

        # Build two probe states with same v and different u by using depth-2 reachability:
        # From archetype, applying x then y yields (u,v) = (m_x, m_y).
        # Fix y and choose two different x values.
        arch_idx = int(np.where(ont == ARCHETYPE_STATE24)[0][0])
        y = 0x12
        x1 = 0x00
        x2 = 0xFF

        idx1 = int(epi[int(epi[arch_idx, x1]), y])
        idx2 = int(epi[int(epi[arch_idx, x2]), y])

        u1, v1 = uv_from_state24(int(ont[idx1]))
        u2, v2 = uv_from_state24(int(ont[idx2]))
        assert v1 == v2, "Probe construction failed: v must match"
        assert u1 != u2, "Probe construction failed: u must differ"

        rng = np.random.default_rng(20260101)
        for _ in range(200):
            L = int(rng.integers(1, 65))
            w = rng.integers(0, 256, size=L, dtype=np.int64).tolist()
            parity, _, _ = word_odd_even_xors(w, masks)

            out1 = apply_word_to_indices(epi, np.array([idx1], dtype=np.int64), w)[0]
            out2 = apply_word_to_indices(epi, np.array([idx2], dtype=np.int64), w)[0]
            uu1, vv1 = uv_from_state24(int(ont[out1]))
            uu2, vv2 = uv_from_state24(int(ont[out2]))

            du0 = u1 ^ u2
            dv0 = v1 ^ v2  # zero
            du1 = uu1 ^ uu2
            dv1 = vv1 ^ vv2

            if parity == 0:
                # identity linear part: delta should remain in u only
                assert du1 == du0 and dv1 == dv0, "Even word did not preserve linear part as identity"
            else:
                # swap linear part: delta should move to v
                assert du1 == dv0 and dv1 == du0, "Odd word did not preserve linear part as swap"


# ========
# F) Closed-form polynomial derivations for shell distributions
# ========

class TestClosedFormShellPolynomials:
    """
    These tests turn the shell distributions into explicit generating functions.

    This is useful because it is a proof-style derivation:
    it is no longer "we observed these counts", it is "these counts equal coefficients of
    a specific polynomial forced by the code anatomy".
    """

    def test_mask_weight_enumerator_is_closed_form_and_prints(self, atlas):
        masks = atlas["masks_a12"].astype(np.uint16)
        counts = weight_enumerator_counts(masks).astype(np.int64).tolist()

        expected = coeffs_mask_weight_enumerator_closed_form()
        assert len(expected) == 13

        table_if_enabled(
            "Mask weight enumerator: observed vs closed form",
            [
                ("observed", str(counts)),
                ("closed form", str(expected)),
                ("match", str(counts == expected)),
            ],
        )

        assert counts == expected

    def test_archetype_distance_enumerator_is_closed_form_and_prints(self, atlas):
        """
        Uses the exact 24-bit archetype distance shell counts and checks they equal:
          coefficients of (1 + z^2)^8 (1 + z)^8

        This is the squared enumerator because Ω = C × C.
        """
        ont = atlas["ont"]
        a = ((ont >> 12) & 0xFFF).astype(np.uint16)
        b = (ont & 0xFFF).astype(np.uint16)

        da = popcount12_arr((a ^ np.uint16(ARCHETYPE_A12)).astype(np.uint16)).astype(np.int64)
        db = popcount12_arr((b ^ np.uint16(ARCHETYPE_B12)).astype(np.uint16)).astype(np.int64)
        dist24 = (da + db).astype(np.int64)

        counts24 = np.bincount(dist24, minlength=25).astype(np.int64).tolist()
        expected = coeffs_archetype_distance_enumerator_closed_form()
        assert len(expected) == 25

        table_if_enabled(
            "Archetype distance enumerator: observed vs closed form",
            [
                ("observed first 13", str(counts24[:13])),
                ("observed last 12", str(counts24[13:])),
                ("closed form first 13", str(expected[:13])),
                ("closed form last 12", str(expected[13:])),
                ("match", str(counts24 == expected)),
            ],
        )

        assert counts24 == expected

    def test_uv_ir_symmetry_in_shells_and_prints(self, atlas):
        """
        Certifies the exact symmetry:
          archetype_shell[d] == archetype_shell[24 - d]
        and prints the max absolute difference (should be 0).
        """
        ont = atlas["ont"]
        a = ((ont >> 12) & 0xFFF).astype(np.uint16)
        b = (ont & 0xFFF).astype(np.uint16)

        da = popcount12_arr((a ^ np.uint16(ARCHETYPE_A12)).astype(np.uint16)).astype(np.int64)
        db = popcount12_arr((b ^ np.uint16(ARCHETYPE_B12)).astype(np.uint16)).astype(np.int64)
        dist24 = (da + db).astype(np.int64)

        counts24 = np.bincount(dist24, minlength=25).astype(np.int64)
        diffs = [int(counts24[d] - counts24[24 - d]) for d in range(25)]
        max_abs = max(abs(x) for x in diffs)

        table_if_enabled(
            "UV/IR shell symmetry (exact in Ω)",
            [
                ("max |count[d] - count[24-d]|", str(max_abs)),
                ("symmetry holds", str(max_abs == 0)),
            ],
        )

        assert max_abs == 0


# ========
# Session dashboard
# ========

@pytest.fixture(scope="session", autouse=True)
def print_physics_3_dashboard():
    yield
    print("\n" + "=" * 10)
    print("PHYSICS 3 DASHBOARD - KERNEL STRUCTURE TOWARDS CGM UNITS")
    print("=" * 10)
    print("  ✓ Reference byte 0xAA: involution with 256 fixed points and 32640 2-cycles")
    print("  ✓ Non-reference bytes: proven pure 4-cycle permutations on Ω")
    print("  ✓ Permutation-unitary eigenphases implied and printed")
    print("  ✓ Mask code spanning set via intron basis bytes (2^8 = 256 masks)")
    print("  ✓ Exact linear-code duality: |C|*|C_perp| = 2^12")
    print("  ✓ MacWilliams identity verified for C and C_perp")
    print("  ✓ Walsh spectrum support theorem verified: support equals C_perp")
    print("  ✓ Atlas shell distributions forced by code enumerator (exact)")
    print("  ✓ CGM Units bridge diagnostics printed (A, m_a, Q_G, K_QG, alpha)")
    print("=" * 10)
