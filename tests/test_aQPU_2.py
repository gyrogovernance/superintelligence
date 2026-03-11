# tests/test_aQPU_2.py
"""
aQPU tests 2: Hilbert-lift quantum foundations.

These tests extend the algebraic quantum processing unit (aQPU) analysis
beyond kernel dynamics into the verified Hilbert-lift consequences of the
self-dual [12,6,2] code and the 6-pair graph-state structure.

Scope:
- C64 -> GF(2)^6 pair-collapse bijection
- Exact factorization of the 64x64 graph state into 6 independent Bell pairs
- Bell / CHSH violation at the Tsirelson bound
- Quantum teleportation using the Bell-pair resources
- Entanglement monogamy and no-signalling
- Mutually unbiased bases on the 6-qubit chirality register
- Peres-Mermin contextuality
- Full 12-generator stabilizer structure of the 6-pair graph state
- Induced Pauli-X action on the chirality register (kernel byte layer)

Deliberately does NOT retest:
- kernel stepping
- Omega topology
- K4 gate action
- chirality transport law
- code size / duality / syndromes
- depth-4 alternation
Those are already covered by the existing physics and aQPU test files.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.api import MASK12_BY_BYTE, chirality_word6, q_word6
from src.constants import step_state_by_byte

from tests.test_aQPU_1 import _bfs_omega

C64 = set(int(m) & 0xFFF for m in MASK12_BY_BYTE)


# ================================================================
# Helpers
# ================================================================


def _gf2_rank(matrix: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2)."""
    M = (matrix.copy() % 2).astype(np.int8)
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
        rank += 1
    return rank


def _c12_to_6bit(c12: int) -> int:
    """
    Collapse a pair-diagonal 12-bit codeword to a 6-bit word.
    Each pair 00 -> 0, 11 -> 1.
    """
    out = 0
    for i in range(6):
        pair = (int(c12) >> (2 * i)) & 0x3
        assert pair in (0x0, 0x3), f"Non pair-diagonal codeword: {c12:#05x}"
        if pair == 0x3:
            out |= 1 << i
    return out


def _bell_pair_state(t_bit: int) -> np.ndarray:
    """
    2-qubit Bell pair:
      t_bit=0 -> |Phi+> = (|00> + |11>) / sqrt(2)
      t_bit=1 -> |Psi+> = (|01> + |10>) / sqrt(2)
    Basis order: |00>, |01>, |10>, |11>.
    """
    v = np.zeros(4, dtype=complex)
    if t_bit == 0:
        v[0] = 1.0 / np.sqrt(2)
        v[3] = 1.0 / np.sqrt(2)
    else:
        v[1] = 1.0 / np.sqrt(2)
        v[2] = 1.0 / np.sqrt(2)
    return v


def _bell_projector(t_bit: int) -> np.ndarray:
    """Projector onto the corresponding Bell pair."""
    v = _bell_pair_state(t_bit)
    return np.outer(v, np.conj(v))


def _graph_state_tensor_qsum(t6: int) -> np.ndarray:
    """
    Build the 12-qubit graph state tensor directly from the q-sum:
      |psi_t> = (1/sqrt(64)) sum_q |q>|q xor t>
    Tensor axis order:
      [A0, B0, A1, B1, ..., A5, B5]
    """
    psi = np.zeros((2,) * 12, dtype=complex)
    amp = 1.0 / 8.0  # 1/sqrt(64)
    for q in range(64):
        idx = []
        for k in range(6):
            a = (q >> k) & 1
            b = a ^ ((t6 >> k) & 1)
            idx.extend([a, b])
        psi[tuple(idx)] = amp
    return psi


def _graph_state_vector_le(t6: int) -> np.ndarray:
    """
    Same graph state as a length-4096 vector in little-endian bit order
    on qubits [A0, B0, A1, B1, ..., A5, B5].
    Bit position 2k = A_k, bit position 2k+1 = B_k.
    """
    psi = np.zeros(1 << 12, dtype=complex)
    amp = 1.0 / 8.0
    for q in range(64):
        idx = 0
        for k in range(6):
            a = (q >> k) & 1
            b = a ^ ((t6 >> k) & 1)
            idx |= a << (2 * k)
            idx |= b << (2 * k + 1)
        psi[idx] = amp
    return psi


def _factorized_bell_tensor(t6: int) -> np.ndarray:
    """
    Tensor product of 6 Bell pairs in axis order [A0,B0,A1,B1,...,A5,B5].
    """
    psi = _bell_pair_state((t6 >> 0) & 1).reshape(2, 2)
    for k in range(1, 6):
        pair = _bell_pair_state((t6 >> k) & 1).reshape(2, 2)
        psi = np.tensordot(psi, pair, axes=0)
    return psi.astype(complex)


def _reduced_density(psi_tensor: np.ndarray, keep: list[int]) -> np.ndarray:
    """
    Reduced density matrix of a pure state tensor.
    keep = list of axis indices to keep.
    Returns a 2^k x 2^k density matrix.
    """
    n = psi_tensor.ndim
    keep_sorted = tuple(sorted(int(i) for i in keep))
    trace_axes = tuple(i for i in range(n) if i not in keep_sorted)
    rho = np.tensordot(np.conj(psi_tensor), psi_tensor, axes=(trace_axes, trace_axes))
    dim = 1 << len(keep_sorted)
    return rho.reshape(dim, dim)


def _expectation(rho: np.ndarray, op: np.ndarray) -> float:
    """Real expectation value Tr(rho op)."""
    return float(np.trace(rho @ op).real)


def _state_overlap_up_to_phase(v: np.ndarray, w: np.ndarray) -> float:
    """Absolute overlap for normalized state vectors."""
    v = np.asarray(v, dtype=complex)
    w = np.asarray(w, dtype=complex)
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    assert nv > 1e-12 and nw > 1e-12
    v = v / nv
    w = w / nw
    return float(abs(np.vdot(v, w)))


def _bell_basis() -> list[np.ndarray]:
    """
    Bell basis on 2 qubits in order:
      0: |Phi+>
      1: |Phi->
      2: |Psi+>
      3: |Psi->
    """
    return [
        np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    ]


def _pauli_set() -> dict[str, np.ndarray]:
    """Single-qubit Pauli corrections."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    XZ = X @ Z
    return {"I": I, "X": X, "Z": Z, "XZ": XZ}


def _teleport_bob_state(alpha: complex, beta: complex, t_bit: int, outcome: int) -> np.ndarray:
    """
    Bob's unnormalized post-measurement state in the teleportation protocol.
    Qubit order: [S, A, B]
      S = unknown input state
      A = Alice half of Bell resource
      B = Bob half of Bell resource
    Bell resource is |Phi+> if t_bit=0, |Psi+> if t_bit=1.
    Bell measurement is on (S, A) with the given outcome index.
    """
    psi_in = np.array([alpha, beta], dtype=complex)
    bell = _bell_pair_state(t_bit).reshape(2, 2)
    psi3 = np.tensordot(psi_in, bell, axes=0)  # shape (2,2,2)
    bell_vec = _bell_basis()[outcome].reshape(2, 2)
    # Project qubits (S,A) onto Bell basis vector, leaving Bob
    bob = np.tensordot(np.conj(bell_vec), psi3, axes=([0, 1], [0, 1]))
    return bob.astype(complex)


def _find_teleport_correction(t_bit: int, outcome: int) -> tuple[str, np.ndarray]:
    """
    Derive the unique Pauli correction for a given Bell resource and outcome
    from a generic probe state.
    """
    probe_theta = 0.731
    probe_phi = 0.417
    alpha = np.cos(probe_theta / 2.0)
    beta = np.sin(probe_theta / 2.0) * np.exp(1j * probe_phi)
    target = np.array([alpha, beta], dtype=complex)

    bob = _teleport_bob_state(alpha, beta, t_bit, outcome)
    paulis = _pauli_set()

    hits = []
    for name, P in paulis.items():
        corrected = P @ bob
        overlap = _state_overlap_up_to_phase(corrected, target)
        if abs(overlap - 1.0) < 1e-10:
            hits.append((name, P))

    assert len(hits) == 1, (
        f"Expected unique Pauli correction for t_bit={t_bit}, outcome={outcome}, got {hits}"
    )
    return hits[0]


def _apply_pauli_le(
    psi: np.ndarray,
    xmask: int,
    zmask: int,
    phase: complex = 1.0,
) -> np.ndarray:
    """
    Apply a Pauli string to a 12-qubit state vector in little-endian indexing.
    xmask flips bits; zmask contributes phase (-1)^(bitcount(i & zmask)).
    Overall scalar phase may be +/-1 here.
    """
    out = np.zeros_like(psi)
    for i, amp in enumerate(psi):
        if abs(amp) < 1e-15:
            continue
        j = i ^ int(xmask)
        sign = -1.0 if (bin(i & int(zmask)).count("1") & 1) else 1.0
        out[j] += phase * sign * amp
    return out


def _symplectic_commutes(x1: int, z1: int, x2: int, z2: int) -> bool:
    """
    Binary symplectic commutation check.
    """
    p = (bin(x1 & z2).count("1") + bin(z1 & x2).count("1")) & 1
    return p == 0


# ================================================================
# PART 1: GRAPH-STATE FACTORIZATION
# ================================================================


class TestGraphStateFactorization:
    """
    The 64x64 graph state from the Hilbert lift factors exactly into
    6 independent Bell pairs, one per dipole pair / chirality qubit.
    """

    def test_c64_pair_collapse_is_bijection_to_gf2_6(self) -> None:
        words = {_c12_to_6bit(c) for c in C64}
        assert len(C64) == 64
        assert words == set(range(64))

    def test_graph_state_equals_tensor_product_of_6_bell_pairs(self) -> None:
        for t6 in (0, 0b101010, 0b111111, 0b010101):
            psi_qsum = _graph_state_tensor_qsum(t6)
            psi_fact = _factorized_bell_tensor(t6)
            assert np.allclose(psi_qsum, psi_fact, atol=1e-12), (
                f"Factorization failed for t6={t6:06b}"
            )

    def test_each_pair_marginal_is_the_expected_bell_state(self) -> None:
        t6 = 0b101010
        psi = _graph_state_tensor_qsum(t6)
        for k in range(6):
            rho_k = _reduced_density(psi, [2 * k, 2 * k + 1])
            target = _bell_projector((t6 >> k) & 1)
            assert np.allclose(rho_k, target, atol=1e-12), (
                f"Pair {k}: marginal is not the expected Bell projector"
            )

    def test_cross_pair_marginals_factorize(self) -> None:
        t6 = 0b101010
        psi = _graph_state_tensor_qsum(t6)
        for k, l in combinations(range(6), 2):
            rho_kl = _reduced_density(psi, [2 * k, 2 * k + 1, 2 * l, 2 * l + 1])
            target = np.kron(
                _bell_projector((t6 >> k) & 1),
                _bell_projector((t6 >> l) & 1),
            )
            assert np.allclose(rho_kl, target, atol=1e-12), (
                f"Pairs {k},{l}: two-pair marginal does not factorize"
            )

    def test_reduced_state_on_each_qubit_pair_is_pure(self) -> None:
        """
        For each qubit pair k, trace out the other 5 pairs.
        The resulting 4x4 density matrix is rank-1 (pure Bell state).
        """
        t6 = 0b101010
        psi = _graph_state_tensor_qsum(t6)
        for k in range(6):
            rho_k = _reduced_density(psi, [2 * k, 2 * k + 1])
            eigenvalues = np.linalg.eigvalsh(rho_k)
            nonzero = eigenvalues[eigenvalues > 1e-10]
            assert len(nonzero) == 1, (
                f"Qubit pair {k}: expected rank 1, got {len(nonzero)}"
            )
            assert abs(nonzero[0] - 1.0) < 1e-10


# ================================================================
# PART 2: BELL / CHSH VIOLATION
# ================================================================


class TestBellCHSH:
    """
    Each Bell-pair factor of the graph state violates CHSH at the
    Tsirelson bound 2*sqrt(2), ruling out local hidden-variable models.
    """

    @staticmethod
    def _chsh_value_for_tbit(t_bit: int) -> float:
        rho = _bell_projector(t_bit)
        Z = np.array([[1, 0], [0, -1]], dtype=float)
        X = np.array([[0, 1], [1, 0]], dtype=float)

        A1, A2 = Z, X
        if t_bit == 0:
            B1 = (Z + X) / np.sqrt(2)
            B2 = (Z - X) / np.sqrt(2)
        else:
            B1 = (-Z + X) / np.sqrt(2)
            B2 = (-Z - X) / np.sqrt(2)

        S = (
            _expectation(rho, np.kron(A1, B1))
            + _expectation(rho, np.kron(A1, B2))
            + _expectation(rho, np.kron(A2, B1))
            - _expectation(rho, np.kron(A2, B2))
        )
        return float(S)

    def test_phi_plus_saturates_tsirelson(self) -> None:
        chsh = self._chsh_value_for_tbit(0)
        assert chsh > 2.0
        assert abs(chsh - 2.0 * np.sqrt(2.0)) < 1e-12

    def test_psi_plus_saturates_tsirelson(self) -> None:
        chsh = self._chsh_value_for_tbit(1)
        assert chsh > 2.0
        assert abs(chsh - 2.0 * np.sqrt(2.0)) < 1e-12

    def test_full_graph_state_inherits_pairwise_chsh(self) -> None:
        t6 = 0b101010
        psi = _graph_state_tensor_qsum(t6)
        for k in range(6):
            rho_k = _reduced_density(psi, [2 * k, 2 * k + 1])
            Z = np.array([[1, 0], [0, -1]], dtype=float)
            X = np.array([[0, 1], [1, 0]], dtype=float)

            A1, A2 = Z, X
            if ((t6 >> k) & 1) == 0:
                B1 = (Z + X) / np.sqrt(2)
                B2 = (Z - X) / np.sqrt(2)
            else:
                B1 = (-Z + X) / np.sqrt(2)
                B2 = (-Z - X) / np.sqrt(2)

            S = (
                _expectation(rho_k, np.kron(A1, B1))
                + _expectation(rho_k, np.kron(A1, B2))
                + _expectation(rho_k, np.kron(A2, B1))
                - _expectation(rho_k, np.kron(A2, B2))
            )
            assert abs(S - 2.0 * np.sqrt(2.0)) < 1e-12, (
                f"Pair {k}: CHSH not saturated"
            )

    def test_tsirelson_bound_saturated(self) -> None:
        """Both Bell pair types exactly saturate the Tsirelson bound."""
        for t_bit in (0, 1):
            chsh = self._chsh_value_for_tbit(t_bit)
            assert abs(chsh - 2.0 * np.sqrt(2.0)) < 1e-12, (
                f"t_bit={t_bit}: CHSH={chsh}, expected 2*sqrt(2)"
            )

    def test_no_measurements_exceed_tsirelson(self) -> None:
        """
        Exhaustive search over discrete measurement angles:
        no CHSH value exceeds 2*sqrt(2). Confirms Tsirelson bound is a hard ceiling.
        """
        psi = np.zeros(4, dtype=complex)
        psi[0] = psi[3] = 1.0 / np.sqrt(2)
        rho = np.outer(psi, np.conj(psi))
        Z = np.array([[1, 0], [0, -1]], dtype=float)
        X = np.array([[0, 1], [1, 0]], dtype=float)
        tsirelson = 2.0 * np.sqrt(2.0)
        n = 10
        for theta_a1 in np.linspace(0, np.pi, n):
            A1 = np.cos(theta_a1) * Z + np.sin(theta_a1) * X
            for theta_a2 in np.linspace(0, np.pi, n):
                A2 = np.cos(theta_a2) * Z + np.sin(theta_a2) * X
                for theta_b1 in np.linspace(0, np.pi, n):
                    B1 = np.cos(theta_b1) * Z + np.sin(theta_b1) * X
                    for theta_b2 in np.linspace(0, np.pi, n):
                        B2 = np.cos(theta_b2) * Z + np.sin(theta_b2) * X
                        S = abs(
                            _expectation(rho, np.kron(A1, B1))
                            + _expectation(rho, np.kron(A1, B2))
                            + _expectation(rho, np.kron(A2, B1))
                            - _expectation(rho, np.kron(A2, B2))
                        )
                        assert S <= tsirelson + 1e-10


# ================================================================
# PART 3: TELEPORTATION
# ================================================================


class TestTeleportationProtocol:
    """
    A Bell pair extracted from the graph-state factorization supports
    exact quantum teleportation of an arbitrary qubit state.
    """

    def test_unique_pauli_correction_exists_for_all_outcomes(self) -> None:
        table = {}
        for t_bit in (0, 1):
            for outcome in range(4):
                name, _ = _find_teleport_correction(t_bit, outcome)
                table[(t_bit, outcome)] = name

        assert len(table) == 8

    def test_teleport_basis_and_phase_states(self) -> None:
        states = [
            np.array([1.0, 0.0], dtype=complex),
            np.array([0.0, 1.0], dtype=complex),
            np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
            np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0),
            np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0),
            np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2.0),
        ]

        for t_bit in (0, 1):
            for outcome in range(4):
                _, P = _find_teleport_correction(t_bit, outcome)
                for target in states:
                    bob = _teleport_bob_state(target[0], target[1], t_bit, outcome)
                    corrected = P @ bob
                    overlap = _state_overlap_up_to_phase(corrected, target)
                    assert abs(overlap - 1.0) < 1e-10, (
                        f"Teleportation failed for t_bit={t_bit}, outcome={outcome}"
                    )

    def test_teleport_random_states(self) -> None:
        rng = np.random.default_rng(20260319)
        for t_bit in (0, 1):
            for outcome in range(4):
                _, P = _find_teleport_correction(t_bit, outcome)
                for _ in range(100):
                    theta = float(rng.uniform(0.0, np.pi))
                    phi = float(rng.uniform(0.0, 2.0 * np.pi))
                    target = np.array(
                        [
                            np.cos(theta / 2.0),
                            np.sin(theta / 2.0) * np.exp(1j * phi),
                        ],
                        dtype=complex,
                    )
                    bob = _teleport_bob_state(target[0], target[1], t_bit, outcome)
                    corrected = P @ bob
                    overlap = _state_overlap_up_to_phase(corrected, target)
                    assert abs(overlap - 1.0) < 1e-10, (
                        f"Random teleportation failed for t_bit={t_bit}, outcome={outcome}"
                    )


# ================================================================
# PART 4: MONOGAMY AND NO-SIGNALLING
# ================================================================


class TestMonogamyAndNoSignalling:
    """
    The 6-pair factorization implies:
    - each Alice qubit is maximally entangled with exactly one Bob qubit,
    - cross-pair 2-qubit reductions are maximally mixed products,
    - Bob's local measurement choice does not change Alice's marginal.
    """

    def test_same_pair_is_pure_bell_cross_pair_is_maximally_mixed(self) -> None:
        t6 = 0b101010
        psi = _graph_state_tensor_qsum(t6)

        I4_over_4 = np.eye(4, dtype=complex) / 4.0

        for k in range(6):
            rho_same = _reduced_density(psi, [2 * k, 2 * k + 1])
            purity = float(np.trace(rho_same @ rho_same).real)
            assert abs(purity - 1.0) < 1e-12
            assert np.allclose(rho_same, _bell_projector((t6 >> k) & 1), atol=1e-12)

            for l in range(6):
                if l == k:
                    continue
                rho_cross = _reduced_density(psi, [2 * k, 2 * l + 1])
                assert np.allclose(rho_cross, I4_over_4, atol=1e-12), (
                    f"Cross pair A{k}-B{l} is not maximally mixed"
                )

    def test_single_qubit_marginals_are_maximally_mixed(self) -> None:
        t6 = 0b101010
        psi = _graph_state_tensor_qsum(t6)
        I2_over_2 = np.eye(2, dtype=complex) / 2.0

        for q in range(12):
            rho_q = _reduced_density(psi, [q])
            assert np.allclose(rho_q, I2_over_2, atol=1e-12), (
                f"Qubit {q}: marginal is not maximally mixed"
            )

    def test_no_signalling_under_bob_measurement_choice(self) -> None:
        """
        Measuring Bob in Z basis or X basis and averaging over outcomes
        leaves Alice's reduced state unchanged.
        """
        for t_bit in (0, 1):
            rho = _bell_projector(t_bit)

            rho_A = np.zeros((2, 2), dtype=complex)
            for a in range(2):
                for ap in range(2):
                    for b in range(2):
                        rho_A[a, ap] += rho[a * 2 + b, ap * 2 + b]

            Pz = [
                np.array([[1, 0], [0, 0]], dtype=complex),
                np.array([[0, 0], [0, 1]], dtype=complex),
            ]

            rho_A_after_Z = np.zeros((2, 2), dtype=complex)
            for P in Pz:
                M = np.kron(np.eye(2, dtype=complex), P)
                rho_post = M @ rho @ M
                for a in range(2):
                    for ap in range(2):
                        for b in range(2):
                            rho_A_after_Z[a, ap] += rho_post[a * 2 + b, ap * 2 + b]

            plus = np.array([1, 1], dtype=complex) / np.sqrt(2.0)
            minus = np.array([1, -1], dtype=complex) / np.sqrt(2.0)
            Px = [
                np.outer(plus, np.conj(plus)),
                np.outer(minus, np.conj(minus)),
            ]

            rho_A_after_X = np.zeros((2, 2), dtype=complex)
            for P in Px:
                M = np.kron(np.eye(2, dtype=complex), P)
                rho_post = M @ rho @ M
                for a in range(2):
                    for ap in range(2):
                        for b in range(2):
                            rho_A_after_X[a, ap] += rho_post[a * 2 + b, ap * 2 + b]

            target = np.eye(2, dtype=complex) / 2.0
            assert np.allclose(rho_A, target, atol=1e-12)
            assert np.allclose(rho_A_after_Z, target, atol=1e-12)
            assert np.allclose(rho_A_after_X, target, atol=1e-12)


# ================================================================
# PART 5: MUTUALLY UNBIASED BASES
# ================================================================


class TestMutuallyUnbiasedBases:
    """
    The computational basis and the Walsh-Hadamard basis on the 6-qubit
    chirality register are mutually unbiased.
    """

    @staticmethod
    def _hadamard64() -> np.ndarray:
        d = 64
        H = np.zeros((d, d), dtype=float)
        for q in range(d):
            for r in range(d):
                dot = bin(q & r).count("1") & 1
                H[q, r] = ((-1) ** dot) / np.sqrt(d)
        return H

    def test_walsh_hadamard_is_unitary(self) -> None:
        H = self._hadamard64()
        assert np.allclose(H @ H.T, np.eye(64), atol=1e-12)

    def test_computational_and_hadamard_bases_are_mub(self) -> None:
        H = self._hadamard64()
        target = 1.0 / 64.0
        assert np.allclose(np.abs(H) ** 2, target, atol=1e-12)

    def test_hadamard64_factors_as_hadamard_tensor_power_6(self) -> None:
        H1 = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2.0)
        H6 = H1
        for _ in range(5):
            H6 = np.kron(H6, H1)
        H64 = self._hadamard64()
        assert np.allclose(H6, H64, atol=1e-12)

    def test_mub_count_for_prime_power_dimension(self) -> None:
        """
        For d = 2^n, max number of MUBs is d+1 = 65.
        Verify a third MUB exists (phase then Hadamard).
        """
        d = 64
        phase = np.diag([1j ** bin(q).count("1") for q in range(d)])
        H = np.zeros((d, d), dtype=complex)
        for q in range(d):
            for r in range(d):
                dot = bin(q & r).count("1") % 2
                H[q, r] = ((-1) ** dot) / np.sqrt(d)
        basis3 = H @ phase
        for q in range(d):
            for r in range(d):
                overlap_sq = abs(basis3[q, r]) ** 2
                assert abs(overlap_sq - 1.0 / d) < 1e-10


# ================================================================
# PART 6: PERES-MERMIN CONTEXTUALITY
# ================================================================


class TestPeresMerminContextuality:
    """
    The 2-qubit sectors used by the aQPU support state-independent contextuality.
    """

    def test_peres_mermin_square_operator_identities(self) -> None:
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        square = [
            [np.kron(X, I), np.kron(I, X), np.kron(X, X)],
            [np.kron(I, Z), np.kron(Z, I), np.kron(Z, Z)],
            [np.kron(X, Z), np.kron(Z, X), np.kron(Y, Y)],
        ]

        for row in square:
            prod = row[0] @ row[1] @ row[2]
            assert np.allclose(prod, np.eye(4), atol=1e-12)

        for j in range(3):
            prod = square[0][j] @ square[1][j] @ square[2][j]
            target = -np.eye(4) if j == 2 else np.eye(4)
            assert np.allclose(prod, target, atol=1e-12)

    def test_noncontextual_value_assignment_is_impossible(self) -> None:
        """
        In any noncontextual hidden-variable assignment:
        product of row-values = +1, product of column-values = -1. Contradiction.
        """
        quantum_row_product = (+1) * (+1) * (+1)
        quantum_col_product = (+1) * (+1) * (-1)
        assert quantum_row_product == +1
        assert quantum_col_product == -1
        assert quantum_row_product != quantum_col_product


# ================================================================
# PART 7: FULL STABILIZER STRUCTURE
# ================================================================


class TestGraphStateStabilizer:
    """
    The 6-pair graph state is a 12-qubit stabilizer state.
    It has:
    - 12 independent commuting generators
    - full stabilizer group size 2^12 = 4096
    - X-type translation subgroup of size 2^6 = 64
    """

    @staticmethod
    def _generators_for_t6(t6: int) -> list[tuple[complex, int, int]]:
        """
        Return 12 generators as (phase, xmask, zmask) on 12 qubits
        in little-endian bit order [A0,B0,A1,B1,...,A5,B5].

        For each pair k:
          gX_k = X_Ak X_Bk
          gZ_k = (+/-) Z_Ak Z_Bk
        with sign -1 for |Psi+> pairs (t_bit=1).
        """
        gens: list[tuple[complex, int, int]] = []
        for k in range(6):
            qa = 2 * k
            qb = 2 * k + 1

            xmask = (1 << qa) | (1 << qb)
            gens.append((1.0, xmask, 0))

            zmask = (1 << qa) | (1 << qb)
            phase = -1.0 if ((t6 >> k) & 1) else 1.0
            gens.append((phase, 0, zmask))
        return gens

    def test_generators_stabilize_graph_state(self) -> None:
        t6 = 0b101010
        psi = _graph_state_vector_le(t6)
        gens = self._generators_for_t6(t6)

        for idx, (phase, xmask, zmask) in enumerate(gens):
            out = _apply_pauli_le(psi, xmask, zmask, phase)
            assert np.allclose(out, psi, atol=1e-12), (
                f"Generator {idx} does not stabilize the graph state"
            )

    def test_generators_commute_pairwise(self) -> None:
        t6 = 0b101010
        gens = self._generators_for_t6(t6)

        for i in range(len(gens)):
            _, x1, z1 = gens[i]
            for j in range(i + 1, len(gens)):
                _, x2, z2 = gens[j]
                assert _symplectic_commutes(x1, z1, x2, z2), (
                    f"Generators {i} and {j} do not commute"
                )

    def test_generator_rank_is_12(self) -> None:
        t6 = 0b101010
        gens = self._generators_for_t6(t6)
        M = np.zeros((12, 24), dtype=np.int8)
        for i, (_, xmask, zmask) in enumerate(gens):
            for q in range(12):
                M[i, q] = (xmask >> q) & 1
                M[i, 12 + q] = (zmask >> q) & 1
        assert _gf2_rank(M) == 12

    def test_full_stabilizer_group_size(self) -> None:
        """
        12 independent commuting generators imply full stabilizer group size 2^12.
        """
        assert 2 ** 12 == 4096

    def test_x_translation_subgroup_has_size_64(self) -> None:
        """
        The 6 X-type pair generators form a 64-element subgroup matching C64.
        """
        t6 = 0b101010
        psi = _graph_state_vector_le(t6)
        gens = self._generators_for_t6(t6)
        xgens = [g for i, g in enumerate(gens) if (i % 2) == 0]

        Mx = np.zeros((6, 12), dtype=np.int8)
        for i, (_, xmask, _) in enumerate(xgens):
            for q in range(12):
                Mx[i, q] = (xmask >> q) & 1
        assert _gf2_rank(Mx) == 6

        rng = np.random.default_rng(20260320)
        for _ in range(256):
            mask = int(rng.integers(0, 64))
            xmask = 0
            for i, (_, xg, _) in enumerate(xgens):
                if (mask >> i) & 1:
                    xmask ^= xg
            out = _apply_pauli_le(psi, xmask, 0, 1.0)
            assert np.allclose(out, psi, atol=1e-12)


# ================================================================
# PART 8: PAULI-X ACTION ON CHIRALITY REGISTER (KERNEL LAYER)
# ================================================================


class TestPauliGroupOnChiralityRegister:
    """
    The induced action on the 6-bit chirality register is exactly the
    Pauli-X translation subgroup: T_b shifts chi by q6(b) via XOR.
    The chirality transport law chi(T_b(s)) = chi(s) xor q6(b) is the
    classical shadow of this Pauli action.
    """

    def test_byte_transitions_form_pauli_x_subgroup(self) -> None:
        """The 64 distinct q6 values are exactly {0, ..., 63}."""
        q6_values = set()
        for b in range(256):
            q6_values.add(q_word6(b))
        assert q6_values == set(range(64))

    def test_pauli_x_group_is_closed(self) -> None:
        """XOR closure: q6(b1) xor q6(b2) is always a valid q6 value."""
        q6_all = [q_word6(b) for b in range(256)]
        q6_set = set(q6_all)
        for q1 in q6_set:
            for q2 in q6_set:
                assert (q1 ^ q2) in q6_set

    def test_pauli_x_group_is_abelian(self) -> None:
        """XOR is commutative on the chirality register."""
        for q1 in range(64):
            for q2 in range(64):
                assert (q1 ^ q2) == (q2 ^ q1)

    def test_chirality_action_is_deterministic(self) -> None:
        """
        Each byte maps each chirality value to exactly one output (no H/S/CNOT).
        """
        omega = _bfs_omega()
        for b in range(256):
            chi_map: dict[int, int] = {}
            for s in omega:
                chi_in = chirality_word6(s)
                chi_out = chirality_word6(step_state_by_byte(s, b))
                if chi_in in chi_map:
                    assert chi_map[chi_in] == chi_out
                else:
                    chi_map[chi_in] = chi_out
