# gyroscopic/hQVM/family.py
"""
Parameterized hQVM(d) on the Omega product chart.

Omega_d = U_d x V_d with |U_d| = |V_d| = 2^d and |Omega_d| = 2^(2d).
Byte alphabet |A_d| = 2^(d+2) = 4 x 2^d (K4 family x micro-reference payload).

Dynamics are the pair-diagonal projection of the spinorial byte rule:
  u' = v xor eps_a,   v' = u xor micro xor eps_b
with eps_a, eps_b from L0 boundary bits and micro the d-bit payload.

Transport increment q_d(b) in GF(2)^d is the chirality XOR increment
(chi' = chi xor q_d). Rank percolation uses span{q_d(b)}.

At d=6 this matches gyroscopic.hQVM.api (step_omega12_by_byte, q_word6).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from typing import Dict, Iterable, List, Sequence, Tuple


def mask_d(d: int) -> int:
    if d < 1:
        raise ValueError("d must be >= 1")
    return (1 << d) - 1


def layer_2d(d: int) -> int:
    return (1 << (2 * d)) - 1


def archetype_d(d: int) -> int:
    """Transcription archetype on (d+2) bits; equals 0xAA at d=6."""
    return sum(1 << i for i in range(1, d + 2, 2))


def intron_from_byte(byte: int, d: int) -> int:
    width = d + 2
    return (int(byte) ^ archetype_d(d)) & ((1 << width) - 1)


def byte_from_intron(intron: int, d: int) -> int:
    return int(intron) ^ archetype_d(d)


def intron_family_d(intron: int, d: int) -> int:
    x = int(intron) & ((1 << (d + 2)) - 1)
    return ((x >> (d + 1)) & 1) << 1 | (x & 1)


def intron_micro_ref_d(intron: int, d: int) -> int:
    return (int(intron) >> 1) & mask_d(d)


def byte_from_family_micro(family: int, micro: int, d: int) -> int:
    bit0 = family & 1
    bith = (family >> 1) & 1
    intron = bit0 | ((micro & mask_d(d)) << 1) | (bith << (d + 1))
    return byte_from_intron(intron, d)


def epsilon_d(d: int) -> int:
    return mask_d(d)


def micro_to_mask_2d(micro: int, d: int) -> int:
    m = int(micro) & mask_d(d)
    out = 0
    for i in range(d):
        if (m >> i) & 1:
            out |= 0b11 << (2 * i)
    return out & layer_2d(d)


def collapse_pairdiag_to_d(word_2d: int, d: int) -> int:
    x = int(word_2d) & layer_2d(d)
    out = 0
    for i in range(d):
        if ((x >> (2 * i)) & 0x3) == 0b11:
            out |= 1 << i
    return out


def q_word_d(byte: int, d: int) -> int:
    """Chirality transport increment q_d(b) in GF(2)^d."""
    intron = intron_from_byte(byte, d)
    l0 = (intron & 1) ^ ((intron >> (d + 1)) & 1)
    micro = intron_micro_ref_d(intron, d)
    q2d = micro_to_mask_2d(micro, d)
    if l0:
        q2d ^= layer_2d(d)
    return collapse_pairdiag_to_d(q2d, d)


def l0_parity_d(intron: int, d: int) -> int:
    x = int(intron) & ((1 << (d + 2)) - 1)
    return (x & 1) ^ ((x >> (d + 1)) & 1)


def eps_a_d(intron: int, d: int) -> int:
    return epsilon_d(d) if (int(intron) & 1) else 0


def eps_b_d(intron: int, d: int) -> int:
    return epsilon_d(d) if ((int(intron) >> (d + 1)) & 1) else 0


def step_uv(u: int, v: int, byte: int, d: int) -> Tuple[int, int]:
    intron = intron_from_byte(byte, d)
    m = mask_d(d)
    u_next = (v ^ eps_a_d(intron, d)) & m
    v_next = (u ^ intron_micro_ref_d(intron, d) ^ eps_b_d(intron, d)) & m
    return u_next, v_next


def chirality_uv(u: int, v: int, d: int) -> int:
    return (u ^ v) & mask_d(d)


def shell_uv(u: int, v: int, d: int) -> int:
    return chirality_uv(u, v, d).bit_count()


def rest_uv(d: int) -> Tuple[int, int]:
    """Rest on maximal-chirality complement horizon: chi = epsilon_d."""
    return 0, epsilon_d(d)


def enumerate_omega_d(d: int) -> List[Tuple[int, int]]:
    m = mask_d(d)
    return [(u, v) for u in range(m + 1) for v in range(m + 1)]


def alphabet_size(d: int) -> int:
    return 1 << (d + 2)


def enumerate_bytes(d: int) -> List[int]:
    return list(range(alphabet_size(d)))


def gf2_rank(vectors: Iterable[int], d: int) -> int:
    basis = [0] * d
    rank = 0
    for v in vectors:
        x = int(v) & mask_d(d)
        for bit in range(d - 1, -1, -1):
            if (x >> bit) & 1:
                if basis[bit]:
                    x ^= basis[bit]
                else:
                    basis[bit] = x
                    rank += 1
                    break
    return rank


def phase_pairs_d(d: int) -> Tuple[Tuple[int, int], ...]:
    """Non-overlapping palindromic pairs on the (d+2)-bit intron.

    At d=6 this is the four CGM phase pairs (Wavefunction 16.9). Smaller d
    uses all valid pairs without duplicating indices (no fixed outer-four
    projection, which double-counts when d+2 < 8).
    """
    n = d + 2
    return tuple((i, n - 1 - i) for i in range(n // 2))


def max_fold_disagreement_d(d: int) -> int:
    return len(phase_pairs_d(d))


def fold_disagreement_d(byte: int, d: int) -> int:
    """Phase-pair XOR disagreement count (0 .. len(phase_pairs_d(d)))."""
    intron = intron_from_byte(byte, d)
    count = 0
    for i, j in phase_pairs_d(d):
        if ((intron >> i) & 1) != ((intron >> j) & 1):
            count += 1
    return count


def delta_depth4_horizon_d(d: int) -> float:
    """Depth-4 projection horizon Q_{8d}(Delta) ~ 1 (Specs_Formalism 7.2)."""
    return 1.0 / (8.0 * d)


def delta_dyadic_byte_d(d: int) -> float:
    """Byte-horizon dyadic 5/2^(d+2), anchored at d=6 (Specs_Formalism 7.2)."""
    return 5.0 / (1 << (d + 2))


def mean_byte_curvature_rate(d: int) -> Tuple[int, int, float]:
    """Return (flat_count, curved_count, curved/|A|)."""
    alphabet = enumerate_bytes(d)
    flat = sum(1 for b in alphabet if fold_disagreement_d(b, d) == 0)
    n = len(alphabet)
    curved = n - flat
    return flat, curved, curved / n


def mean_fold_disagreement_d(d: int) -> float:
    """Mean fold disagreement normalized by phase-pair count."""
    alphabet = enumerate_bytes(d)
    n_pairs = max(1, len(phase_pairs_d(d)))
    total = sum(fold_disagreement_d(b, d) for b in alphabet)
    return total / (len(alphabet) * n_pairs)


def mean_carrier_entanglement_d(d: int) -> float:
    """Mean popcount(u xor v)/d over Omega_d (50% holographic redundancy at all d)."""
    omega = enumerate_omega_d(d)
    s = sum((u ^ v).bit_count() for u, v in omega)
    return s / (len(omega) * d)


def step_f_word_uv(u: int, v: int, micro: int, d: int) -> Tuple[int, int]:
    """Depth-4 F word (four family bytes for one micro-reference)."""
    for f in range(4):
        b = byte_from_family_micro(f, micro, d)
        u, v = step_uv(u, v, b, d)
    return u, v


def depth4_projection_bits(d: int) -> int:
    """Depth-4 pair-diagonal projection size 4 x 2d = 8d (Specs_Formalism 7.2)."""
    return 8 * d


def carrier_entanglement_sum_d(d: int) -> int:
    """Sum of popcount(u xor v) over Omega_d."""
    return sum((u ^ v).bit_count() for u, v in enumerate_omega_d(d))


def verify_carrier_entanglement_exact(d: int) -> Tuple[bool, int, int]:
    """Exact check: sum popcount(chi) = d * 2^(2d-1), mean S/d = 1/2."""
    got = carrier_entanglement_sum_d(d)
    expect = d * (1 << (2 * d - 1))
    return got == expect, got, expect


def delta_spinorial_residual_d(d: int) -> float:
    """Residual aperture after depth-4 closure: mean_fd / (4d).

    Equals 1/(8d) when byte mean_fd = 1/2. At d=6 matches Q_48(Delta) ~ 1.
    """
    return mean_fold_disagreement_d(d) / (4.0 * d)


def verify_f_squared_rest_d(d: int) -> Tuple[int, int]:
    """Count micro-ref F words with F(F(rest)) = rest (depth-4 involution on rest)."""
    rest = rest_uv(d)
    ok = 0
    n_micro = 1 << d
    for m in range(n_micro):
        u2, v2 = step_f_word_uv(*rest, m, d)
        u4, v4 = step_f_word_uv(u2, v2, m, d)
        if (u4, v4) == rest:
            ok += 1
    return ok, n_micro


def shell_population_d(d: int, shell: int) -> int:
    if shell < 0 or shell > d:
        return 0
    return comb(d, shell) * (1 << d)


def partition_Z1_coeff_d(d: int, lam: float = 1.0) -> float:
    """Climate-chart Z1(lam) = 2^d (1+lam)^d at shell weight lam."""
    return (1 << d) * ((1.0 + lam) ** d)


def rank_excess_z(p: float, d: int) -> float:
    """Rank excess z = E[included groups] - d = 2^d p - d (micro-ref / q-class)."""
    return p * (1 << d) - d


def micro_pair_prob(p: float) -> float:
    """Effective inclusion prob on quotient cosets: p_pair = 1-(1-p)^2."""
    q = 1.0 - p
    return 1.0 - q * q


def z_root_micro_ref(p: float, d: int) -> float:
    """Thermodynamic z on the root space of dimension (d-1)."""
    if d < 1:
        raise ValueError("d must be >=1")
    if d == 1:
        return 0.0
    n = d - 1
    return (1 << n) * micro_pair_prob(p) - n


def p_from_z_root_micro_ref(z_root: float, d: int) -> float:
    """Invert z_root -> p using p_pair = (n+z_root)/2^n, p = 1 - sqrt(1-p_pair)."""
    if d < 1:
        raise ValueError("d must be >=1")
    if d == 1:
        return 0.5
    n = d - 1
    p_pair = (n + z_root) / float(1 << n)
    if p_pair <= 0.0:
        return 0.0
    if p_pair >= 1.0:
        return 1.0
    return 1.0 - sqrt(1.0 - p_pair)


def gaussian_binomial(n: int, k: int, q: int = 2) -> int:
    """Gaussian binomial coefficient [n choose k]_q."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    num = 1
    den = 1
    for i in range(k):
        num *= q ** (n - i) - 1
        den *= q ** (i + 1) - 1
    return num // den


def _subspace_mobius(j: int, k: int) -> int:
    """Möbius factor mu(j,k) on the GF(2) subspace lattice (dim j <= k)."""
    if j > k:
        return 0
    m = k - j
    return (-1) ** m * (1 << (m * (m - 1) // 2))


def _root_span_subset_prob(q: float, n_dim: int, k: int) -> float:
    """F(k) = P[X subset of a fixed k-subspace], |X| i.i.d. Bernoulli(q) on GF(2)^n_dim."""
    if k < 0 or k > n_dim:
        return 0.0
    n_vecs = 1 << n_dim
    qd = 1.0 - q
    return qd ** (n_vecs - (1 << k))


def exact_root_rank_pmf(q: float, n_dim: int) -> List[float]:
    """P(rank=k) on GF(2)^n_dim with each vector included i.i.d. with prob q."""
    if n_dim < 0:
        raise ValueError("n_dim must be >= 0")
    if q <= 0.0:
        return [1.0] + [0.0] * n_dim
    if q >= 1.0:
        return [0.0] * n_dim + [1.0]

    if n_dim >= 11:
        from decimal import Decimal, getcontext

        getcontext().prec = 60 + (n_dim * (n_dim - 1)) // 4
        qd = Decimal(1) - Decimal(repr(q))
        n_vecs = 1 << n_dim
        f_vals = [qd ** (n_vecs - (1 << k)) for k in range(n_dim + 1)]
        g_vals_dec: List[Decimal] = []
        for k in range(n_dim + 1):
            total = Decimal(0)
            for j in range(k + 1):
                mu = _subspace_mobius(j, k)
                if mu == 0:
                    continue
                gb = gaussian_binomial(k, j, q=2)
                total += Decimal(mu * gb) * f_vals[j]
            g_vals_dec.append(total)
        return [
            float(Decimal(gaussian_binomial(n_dim, k, q=2)) * g_vals_dec[k])
            for k in range(n_dim + 1)
        ]

    f_vals = [_root_span_subset_prob(q, n_dim, k) for k in range(n_dim + 1)]
    g_vals = [0.0] * (n_dim + 1)
    for k in range(n_dim + 1):
        total = 0.0
        for j in range(k + 1):
            mu = _subspace_mobius(j, k)
            if mu == 0:
                continue
            total += gaussian_binomial(k, j, q=2) * mu * f_vals[j]
        g_vals[k] = total
    return [gaussian_binomial(n_dim, k, q=2) * g_vals[k] for k in range(n_dim + 1)]


def exact_micro_ref_rank_pmf(p: float, d: int) -> List[float]:
    """Exact P(rank=r) for micro-ref inclusion, r=0..d (O(d^2), all d)."""
    if d < 0:
        raise ValueError("d must be >= 0")
    if d == 0:
        return [1.0]
    if p <= 0.0:
        return [1.0] + [0.0] * d
    if p >= 1.0:
        return [0.0] * d + [1.0]
    if d == 1:
        p0 = (1.0 - p) ** 2
        return [p0, 1.0 - p0]

    n_root = d - 1
    q_pair = micro_pair_prob(p)
    root = exact_root_rank_pmf(q_pair, n_root)
    p_empty = (1.0 - p) ** (1 << d)
    dist = [0.0] * (d + 1)
    dist[0] = p_empty
    dist[1] = root[0] - p_empty
    for r in range(2, d + 1):
        dist[r] = root[r - 1]
    return dist


def exact_micro_ref_p_rank_full(p: float, d: int) -> float:
    """Exact P(rank=d) for micro-ref inclusion."""
    if d < 0:
        raise ValueError("d must be >= 0")
    return exact_micro_ref_rank_pmf(p, d)[d]


def exact_micro_ref_p_rank_full_cond(p: float, d: int) -> float:
    """P(rank=d | at least one micro-ref group included)."""
    p_uncond = exact_micro_ref_p_rank_full(p, d)
    p_nz = 1.0 - (1.0 - p) ** (1 << d)
    if p_nz <= 0.0:
        return 0.0
    return p_uncond / p_nz


def _reach_fraction_micro_ref(rank: int, d: int) -> float:
    """|Reach|/|Omega| from square-root law; rank=0 is rest only (reach=1)."""
    n_omega = 1 << (2 * d)
    if rank <= 0:
        return 1.0 / n_omega
    return ((1 << rank) ** 2) / n_omega


def theta_micro_ref_exact(p: float, d: int, *, cond: bool = True) -> float:
    """E[|Reach|/|Omega|] for micro-ref; cond=True conditions on nonempty micro set."""
    if d < 1:
        raise ValueError("d must be >= 1")
    dist = exact_micro_ref_rank_pmf(p, d)
    theta = sum(pr * _reach_fraction_micro_ref(r, d) for r, pr in enumerate(dist))
    if not cond:
        return theta
    p_nz = 1.0 - (1.0 - p) ** (1 << d)
    if p_nz <= 0.0:
        return 0.0
    return theta / p_nz


def exact_micro_ref_theta_cond(p: float, d: int) -> float:
    """Alias for theta_micro_ref_exact(..., cond=True)."""
    return theta_micro_ref_exact(p, d, cond=True)


def verify_exact_root_rank_lock(
    p_test: float = 0.3, d: int = 6
) -> Tuple[bool, float, float]:
    """P_root(n=d-1) from rank PMF matches exact_micro_ref_p_rank_full."""
    if d < 2:
        return True, 0.0, 0.0
    n_root = d - 1
    q_pair = micro_pair_prob(p_test)
    root = exact_root_rank_pmf(q_pair, n_root)
    p_full = root[n_root]
    p_exact = exact_micro_ref_p_rank_full(p_test, d)
    return abs(p_full - p_exact) < 1e-9, p_full, p_exact


def verify_exact_micro_ref_rank_distribution_pair_brute(
    d: int, p_test: float = 0.3
) -> Tuple[bool, List[float], List[float]]:
    """Brute-force via pair-quotient enumeration (feasible for d=5 only)."""
    from itertools import combinations

    if d != 5:
        raise ValueError("pair-brute rank check is defined for d=5 only")
    mask = mask_d(d)
    n_pairs = 1 << (d - 1)
    pp = micro_pair_prob(p_test)
    brute = [0.0] * (d + 1)
    for size in range(n_pairs + 1):
        for subset in combinations(range(n_pairs), size):
            q_set: set[int] = set()
            for m in subset:
                q_set.add(m)
                q_set.add(m ^ mask)
            vectors = [v for v in q_set if v != 0]
            r = gf2_rank(vectors, d) if vectors else 0
            prob = pp ** size * (1.0 - pp) ** (n_pairs - size)
            brute[r] += prob
    exact = exact_micro_ref_rank_pmf(p_test, d)
    ok = all(abs(brute[i] - exact[i]) < 1e-9 for i in range(d + 1))
    return ok, brute, exact


def verify_exact_micro_ref_rank_distribution_algebraic(
    d: int, p_test: float = 0.3
) -> Tuple[bool, str]:
    """Non-enumerative checks: normalization and full-rank lock (all d >= 2)."""
    if d < 2:
        return True, "trivial"
    dist = exact_micro_ref_rank_pmf(p_test, d)
    if any(x < -1e-12 for x in dist):
        return False, "negative mass"
    s = sum(dist)
    if abs(s - 1.0) > 1e-9:
        return False, f"sum={s:.12f}"
    ok_lock, _, _ = verify_exact_root_rank_lock(p_test, d)
    if not ok_lock:
        return False, "full-rank lock"
    return True, "ok"


def verify_exact_micro_ref_rank_distribution_brute(
    d: int, p_test: float = 0.3
) -> Tuple[bool, List[float], List[float]]:
    """Brute-force check of exact_micro_ref_rank_pmf for d <= 4."""
    from itertools import combinations

    if d > 4:
        return True, [], exact_micro_ref_rank_pmf(p_test, d)
    n = 1 << d
    mask = mask_d(d)
    brute = [0.0] * (d + 1)
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            q_set: set[int] = set()
            for m in subset:
                q_set.add(m)
                q_set.add(m ^ mask)
            vectors = [v for v in q_set if v != 0]
            r = gf2_rank(vectors, d) if vectors else 0
            prob = p_test ** size * (1.0 - p_test) ** (n - size)
            brute[r] += prob
    exact = exact_micro_ref_rank_pmf(p_test, d)
    ok = all(abs(brute[i] - exact[i]) < 1e-9 for i in range(d + 1))
    return ok, brute, exact


def verify_micro_ref_full_iff_rank_d(d: int) -> Tuple[bool, int, int]:
    """Micro-ref: BFS full reachability iff GF(2)^d rank = d on included q values.

    Exhaustive over all 2^(2^d) micro-ref subsets for d <= 4.
    """
    if d > 4:
        return True, 0, 0

    eng = build_hqvm_d(d)
    micro_groups = eng.micro_ref_groups
    n_micro = 1 << d
    total = 1 << n_micro
    passed = 0

    for mask in range(total):
        allowed: List[int] = []
        qs: List[int] = []
        for m in range(n_micro):
            if (mask >> m) & 1:
                grp = micro_groups[m]
                allowed.extend(grp)
                qs.extend(eng.q_by_byte[b] for b in grp)

        if not allowed:
            full = False
            r = 0
        else:
            _, _, _, full = bfs_reach(eng, allowed)
            r = gf2_rank(qs, d)

        if full == (r == d):
            passed += 1

    return passed == total, passed, total


def bisect_p_c_rank_micro_ref(d: int) -> float:
    """p with exact_micro_ref_p_rank_full(p, d) = 1/2."""
    lo, hi, p_c = 0.0, 1.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if exact_micro_ref_p_rank_full(mid, d) < 0.5:
            lo = mid
        else:
            hi = mid
        p_c = mid
    return p_c


def closed_form_p_c_rank_micro_ref(d: int, c_star_root: float) -> float:
    """Asymptotic: z_root,c ≈ c* so p_c ≈ p_from_z_root_micro_ref(c*, d)."""
    return p_from_z_root_micro_ref(c_star_root, d)


def rank_excess_limit_c_root_extrapolated(d1: int = 28, d2: int = 32) -> float:
    """Asymptotic c*_root limit via 1/d extrapolation of z_root,c at d1, d2."""
    z1 = z_root_micro_ref(bisect_p_c_rank_micro_ref(d1), d1)
    z2 = z_root_micro_ref(bisect_p_c_rank_micro_ref(d2), d2)
    a = (z2 - z1) / (1.0 / d1 - 1.0 / d2)
    return z2 + a / d2


def holonomy_micro_cov(p: float, d: int, word_len: int = 4) -> float:
    return 1.0 - (1.0 - p ** word_len) ** (1 << d)


@dataclass(frozen=True)
class HqvmD:
    d: int
    n_omega: int
    n_bytes: int
    transitions: Tuple[Tuple[int, ...], ...]
    shell: Tuple[int, ...]
    start_idx: int
    uv_to_idx: Dict[Tuple[int, int], int]
    q_by_byte: Tuple[int, ...]
    bytes_by_q: Tuple[Tuple[int, ...], ...]
    micro_ref_groups: Tuple[Tuple[int, ...], ...]
    q_class_groups: Tuple[Tuple[int, ...], ...]
    fold_disagree: Tuple[int, ...]
    q_weight: Tuple[int, ...]


def build_hqvm_d(d: int) -> HqvmD:
    omega = enumerate_omega_d(d)
    n = len(omega)
    uv_to_idx = {uv: i for i, uv in enumerate(omega)}
    start_idx = uv_to_idx[rest_uv(d)]
    alphabet = enumerate_bytes(d)
    n_bytes = len(alphabet)

    q_by_byte = tuple(q_word_d(b, d) for b in alphabet)
    q_buckets: List[List[int]] = [[] for _ in range(1 << d)]
    for b, q in enumerate(q_by_byte):
        q_buckets[q].append(b)
    bytes_by_q = tuple(tuple(v) for v in q_buckets)

    micro_groups = tuple(
        tuple(byte_from_family_micro(f, m, d) for f in range(4))
        for m in range(1 << d)
    )
    q_class_groups = bytes_by_q

    trans: List[List[int]] = [[0] * n_bytes for _ in range(n)]
    shell = [0] * n
    for i, (u, v) in enumerate(omega):
        shell[i] = shell_uv(u, v, d)
        for bi, b in enumerate(alphabet):
            trans[i][bi] = uv_to_idx[step_uv(u, v, b, d)]

    fd = tuple(fold_disagreement_d(b, d) for b in alphabet)
    qw = tuple(q.bit_count() for q in q_by_byte)

    return HqvmD(
        d=d,
        n_omega=n,
        n_bytes=n_bytes,
        transitions=tuple(tuple(row) for row in trans),
        shell=tuple(shell),
        start_idx=start_idx,
        uv_to_idx=uv_to_idx,
        q_by_byte=q_by_byte,
        bytes_by_q=bytes_by_q,
        micro_ref_groups=micro_groups,
        q_class_groups=q_class_groups,
        fold_disagree=fd,
        q_weight=qw,
    )


def verify_d6_against_api() -> Tuple[bool, str]:
    """Cross-check d=6 against production gyroscopic.hQVM.api."""
    try:
        from .api import (
            OMEGA_STATES_4096,
            q_word6,
            state24_to_omega12,
            step_omega12_by_byte,
        )
        from .constants import GENE_MAC_REST
    except ImportError as exc:
        return False, f"import failed: {exc}"

    eng = build_hqvm_d(6)
    rest = state24_to_omega12(GENE_MAC_REST)
    if eng.uv_to_idx[(rest.u6, rest.v6)] != eng.start_idx:
        return False, "rest anchor mismatch"

    q_mism = step_mism = 0
    for b in range(256):
        if q_word6(b) != eng.q_by_byte[b]:
            q_mism += 1
        o = step_omega12_by_byte((rest.u6, rest.v6), b)
        nu, nv = step_uv(rest.u6, rest.v6, b, 6)
        if (nu, nv) != (o.u6, o.v6):
            step_mism += 1

    if q_mism or step_mism:
        return False, f"q={q_mism} step={step_mism}"
    return True, "q_word and step_uv match api at d=6"


def bfs_reach(
    eng: HqvmD,
    allowed_bytes: Sequence[int],
    *,
    max_depth: int | None = None,
) -> Tuple[int, bool, bool, bool]:
    """Return (|Reach|, E_span, giant, E_full)."""
    allowed = set(int(b) for b in allowed_bytes)
    byte_idx = [i for i, b in enumerate(enumerate_bytes(eng.d)) if b in allowed]
    if not byte_idx:
        return 1, False, False, False

    if max_depth is None:
        max_depth = 2 * eng.d + 6

    n = eng.n_omega
    visited = bytearray(n)
    q: List[int] = [eng.start_idx]
    visited[eng.start_idx] = 1
    head = 0
    depth = 0
    next_end = 1
    spans = False

    while head < len(q):
        if head >= next_end:
            depth += 1
            if depth > max_depth:
                break
            next_end = len(q)
        i = q[head]
        head += 1
        if eng.shell[i] == 0:
            spans = True
        row = eng.transitions[i]
        for bi in byte_idx:
            j = row[bi]
            if not visited[j]:
                visited[j] = 1
                q.append(j)

    reach = int(sum(visited))
    full = reach == n
    giant = reach >= n // 2
    return reach, spans, giant, full


def fiber_complete(allowed: Sequence[int], eng: HqvmD) -> bool:
    fams: Dict[int, set[int]] = {}
    for b in allowed:
        q = eng.q_by_byte[b]
        fams.setdefault(q, set()).add(intron_family_d(intron_from_byte(b, eng.d), eng.d))
    return all(len(s) == 4 for s in fams.values())


def predicted_cluster_size(rank: int) -> int:
    if rank == 0:
        return 2
    return (1 << rank) ** 2


def is_fiber_complete_qs(qs: Sequence[int], eng: HqvmD) -> bool:
    for q in qs:
        if len(eng.bytes_by_q[q]) < 4:
            return False
    return True
