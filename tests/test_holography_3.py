# tests/test_holography_3.py
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.router.constants import (
    ARCHETYPE_A12,
    LAYER_MASK_12,
    byte_to_intron,
    expand_intron_to_mask24,
    mask12_for_byte,
    unpack_state,
)
from src.router.kernel import RouterKernel

SEP = "=========="

# -----------------------------
# Basic helpers
# -----------------------------

def popcount12(x: int) -> int:
    return bin(x & 0xFFF).count("1")

def dot_parity(x: int, y: int) -> int:
    return bin(x & y).count("1") & 1

def horizon_indices(ont: NDArray[np.uint32]) -> list[int]:
    out = []
    for i, s in enumerate(ont):
        a, b = unpack_state(int(s))
        if a == (b ^ LAYER_MASK_12):
            out.append(i)
    return out

def build_vertex_map(ont: NDArray[np.uint32], h_idxs: list[int]) -> dict[int, int]:
    """
    Same vertex map as holography_2:
    horizon A -> vertex in {0,1,2,3}.
    """
    vertex_map: dict[int, int] = {}
    for h_idx in h_idxs:
        a, _ = unpack_state(int(ont[h_idx]))
        frame0 = a & 0x3F
        frame1 = (a >> 6) & 0x3F
        frame_parity = (popcount12(frame0) + popcount12(frame1)) % 2

        row0_bits = (a & 0x003) | ((a >> 4) & 0x0C0)
        row1_bits = ((a >> 2) & 0x003) | ((a >> 6) & 0x0C0)
        row2_bits = ((a >> 4) & 0x003) | ((a >> 8) & 0x0C0)
        row_sum = popcount12(row0_bits) + popcount12(row1_bits) + popcount12(row2_bits)
        row_parity = row_sum % 2

        vertex = (frame_parity * 2) + row_parity
        vertex_map[a] = vertex
    return vertex_map

def build_charge_by_mask(
    ont: NDArray[np.uint32],
    epi: NDArray[np.uint32],
    h_idxs: list[int],
    vertex_map: dict[int, int],
) -> dict[int, int]:
    """
    Same as holography_2: charge_by_mask[m] is the 2-bit vertex XOR shift induced by byte^2 on horizon.
    """
    from collections import Counter

    charge_by_mask: dict[int, int] = {}
    unique_masks = set(mask12_for_byte(b) for b in range(256))
    mask_to_any_byte: dict[int, int] = {}
    for b in range(256):
        m = mask12_for_byte(b)
        if m not in mask_to_any_byte:
            mask_to_any_byte[m] = b

    for m in unique_masks:
        b = mask_to_any_byte[m]
        charges_seen = []
        for h_idx in h_idxs:
            a0, _ = unpack_state(int(ont[h_idx]))
            v0 = vertex_map[a0]
            h1_idx = int(epi[h_idx, b])
            h2_idx = int(epi[h1_idx, b])
            a2, b2 = unpack_state(int(ont[h2_idx]))
            if a2 != (b2 ^ LAYER_MASK_12):
                continue
            v2 = vertex_map[a2]
            charges_seen.append(v0 ^ v2)

        c = Counter(charges_seen)
        mc, cnt = c.most_common(1)[0]
        assert cnt == len(charges_seen), f"charge not constant for mask 0x{m:03x}: {c}"
        charge_by_mask[m] = mc

    return charge_by_mask

def solve_q0_q1(charge_by_mask: dict[int, int]) -> tuple[int, int]:
    """
    Solve parity checks q0,q1 such that for all m in C:
      chi(m) = (<q0,m>, <q1,m>)
    matches charge_by_mask.
    """
    C = {mask12_for_byte(b) for b in range(256)}
    C_list = sorted(C)

    t0 = {m: charge_by_mask[m] & 1 for m in C}
    t1 = {m: (charge_by_mask[m] >> 1) & 1 for m in C}

    rhs0 = [t0[m] for m in C_list]
    rhs1 = [t1[m] for m in C_list]

    def solve_one(rhs: list[int]) -> int:
        for q in range(4096):
            ok = True
            for m, bit in zip(C_list, rhs):
                if dot_parity(q, m) != (bit & 1):
                    ok = False
                    break
            if ok:
                return q
        raise AssertionError("no solution for q")

    return solve_one(rhs0), solve_one(rhs1)

def chi_from_q(q0: int, q1: int, m: int) -> int:
    b0 = dot_parity(q0, m)
    b1 = dot_parity(q1, m)
    return b0 + (b1 << 1)

def gf2_rank(M: np.ndarray) -> int:
    """GF(2) rank by Gaussian elimination."""
    A = (M.astype(np.uint8) & 1).copy()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot], :] = A[[pivot, r], :]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        r += 1
        if r == rows:
            break
    return r

def build_generator_matrix_G() -> NDArray[np.uint8]:
    """
    Build generator matrix G (12×8) from intron basis bytes:
      b_i = 0xAA XOR (1<<i)
    Each column is the 12-bit mask for that byte.
    """
    gen_bytes = [((1 << i) ^ 0xAA) & 0xFF for i in range(8)]
    G = np.zeros((12, 8), dtype=np.uint8)
    for col in range(8):
        intron = byte_to_intron(gen_bytes[col])
        mask24 = expand_intron_to_mask24(intron)
        mask12 = (mask24 >> 12) & 0xFFF
        for bit in range(12):
            if (mask12 >> bit) & 1:
                G[bit, col] = 1
    return G

def ambiguity_subcode_from_erasure(G: NDArray[np.uint8], erased_bits: set[int]) -> set[int]:
    """
    Compute ambiguity subcode E_S:
      E_S = { c in C : c has zeros on all observed positions }
    by enumerating message vectors x in F2^8 and mapping to codewords c=Gx.
    """
    observed = [i for i in range(12) if i not in erased_bits]
    E: set[int] = set()
    for x_msg in range(256):
        c = 0
        for i in range(8):
            if (x_msg >> i) & 1:
                for bit in range(12):
                    if G[bit, i]:
                        c ^= (1 << bit)
        ok = True
        for bit_pos in observed:
            if (c >> bit_pos) & 1:
                ok = False
                break
        if ok:
            E.add(c & 0xFFF)
    return E

# ========
# H3-1: Frame0 projection is uniform 4-to-1 over the mask code C
# ========

def test_h3_frame0_projection_uniform_4_to_1():
    """
    DECISIVE: The frame0 projection (low 6 bits) of the mask code is uniform 4-to-1.

    For each m in C, define frame0(m) = m & 0x3F (6 bits).
    Then:
      - every 6-bit pattern occurs
      - each occurs exactly 4 times in C
      - the 4 preimages differ only in bits 6 and 7 (i.e., only x6,x7 vary)

    This is the cleanest single-step formalization of "6-bit lossy micro-reference".
    """
    C = sorted({mask12_for_byte(b) for b in range(256)})
    buckets: dict[int, list[int]] = {}
    for m in C:
        f0 = m & 0x3F
        buckets.setdefault(f0, []).append(m)

    assert len(buckets) == 64, f"expected all 64 frame0 values, got {len(buckets)}"
    for f0, ms in buckets.items():
        assert len(ms) == 4, f"frame0={f0:02x} expected 4 preimages, got {len(ms)}"
        # preimages must share bits 0..5 and 8..11; vary only bits 6..7
        base = ms[0]
        for m in ms[1:]:
            diff = (m ^ base) & 0xFFF
            assert diff in (0x040, 0x080, 0x0C0), f"frame0={f0:02x} preimages differ outside bits6-7: diff=0x{diff:03x}"

    print(SEP)
    print("H3: frame0 projection uniformity")
    print(SEP)
    print("  verified: 64 buckets × 4 masks each, differences only in bits 6 and 7")

# ========
# H3-2: Exhaustive erasure taxonomy (size-4 erasures) by (rank, ambiguity, chi-rank)
# ========

def test_h3_exhaustive_erasure_taxonomy_size4():
    """
    DECISIVE: Exhaustive classification of all 4-bit erasures (C(12,4)=495).

    For each erased set E of size 4:
      - compute rank(G_S) for observed bits S = complement(E)
      - ambiguity size |E_S| = 2^(8 - rank(G_S))
      - compute chi-rank loss category: rank(chi(E_S)) in {0,1,2}

    This produces a complete "holographic redundancy atlas" for 4-bit erasures.
    """
    atlas_dir = Path("data/atlas")
    K = RouterKernel(atlas_dir)
    ont = K.ontology
    epi = K.epistemology

    h_idxs = horizon_indices(ont)
    vertex_map = build_vertex_map(ont, h_idxs)
    charge_by_mask = build_charge_by_mask(ont, epi, h_idxs, vertex_map)
    q0, q1 = solve_q0_q1(charge_by_mask)

    G = build_generator_matrix_G()

    def chi(m: int) -> int:
        return chi_from_q(q0, q1, m)

    def chi_rank(chis: set[int]) -> int:
        return {1: 0, 2: 1, 4: 2}[len(chis)]

    counts: dict[tuple[int, int, int], int] = {}
    # key = (rank_GS, ambiguity_size, chi_rank)

    all_bits = list(range(12))
    for erased in combinations(all_bits, 4):
        erased_set = set(erased)
        observed = [i for i in all_bits if i not in erased_set]
        rank_GS = gf2_rank(G[np.array(observed, dtype=np.intp), :])
        ambiguity = 2 ** (8 - rank_GS)
        E = ambiguity_subcode_from_erasure(G, erased_set)
        chis = {chi(e) for e in E}
        k = (rank_GS, ambiguity, chi_rank(chis))
        counts[k] = counts.get(k, 0) + 1

    total = sum(counts.values())
    assert total == 495, f"expected 495 erasures, got {total}"

    # Hard structural sanity: ranks cannot exceed 8
    assert all(r <= 8 for (r, _, _) in counts.keys())

    print(SEP)
    print("H3: exhaustive size-4 erasure taxonomy")
    print(SEP)
    for key in sorted(counts.keys()):
        r, amb, cr = key
        print(f"  rank(G_S)={r}, ambiguity={amb:3d}, chi-rank={cr} : count={counts[key]}")

# ========
# H3-3: Minimal information-set size (how many observed bits are needed for unique decode)
# ========

def test_h3_min_information_set_size_is_8():
    """
    DECISIVE: Find the minimum number of observed bit positions needed for rank 8.

    For each subset S ⊆ {0..11} of observed bits, compute rank(G_S).
    Let r_max(s) be the maximum rank among all |S|=s subsets.
    Then the minimal s for which r_max(s)=8 is the information-set threshold.

    Expectation (from kernel structure): min observed bits = 8.
    """
    G = build_generator_matrix_G()

    rmax_by_size = {s: 0 for s in range(13)}

    # Iterate all subsets via bitmasks (0..4095)
    for mask in range(1 << 12):
        S = [i for i in range(12) if (mask >> i) & 1]
        s = len(S)
        if s == 0:
            r = 0
        else:
            r = gf2_rank(G[np.array(S, dtype=np.intp), :])
        if r > rmax_by_size[s]:
            rmax_by_size[s] = r

    min_s = min(s for s in range(13) if rmax_by_size[s] == 8)

    print(SEP)
    print("H3: information-set threshold")
    print(SEP)
    for s in range(13):
        print(f"  |observed|={s:2d}: max rank = {rmax_by_size[s]}")
    print(f"  minimal |observed| with rank 8: {min_s}")

    assert min_s == 8, f"expected threshold 8, got {min_s}"

# ========
# H3-4: Boundary stabilizer subgroup D0 = ker(chi) and vertex coset reconstruction
# ========

def test_h3_boundary_stabilizer_subgroup_and_vertex_cosets():
    """
    DECISIVE: The K4 vertex classes are exactly cosets of D0 = {m in C : chi(m)=0}.

    1) Compute chi from q0,q1 (as in holography_2).
    2) Define D0 = ker(chi) inside the mask code C.
       - Expect |D0|=64 and rank(D0)=6.
    3) Let U_v be the set of horizon u-coordinates for vertex v (size 64).
       Prove U_v is a coset of D0:
         U_v = u0 XOR D0 for any u0 in U_v.

    This is the strongest “boundary internal reconstruction” theorem you have:
    from one horizon representative + the stabilizer subcode you recover the entire vertex boundary region.
    """
    atlas_dir = Path("data/atlas")
    K = RouterKernel(atlas_dir)
    ont = K.ontology
    epi = K.epistemology

    h_idxs = horizon_indices(ont)
    vertex_map = build_vertex_map(ont, h_idxs)
    charge_by_mask = build_charge_by_mask(ont, epi, h_idxs, vertex_map)
    q0, q1 = solve_q0_q1(charge_by_mask)

    C = sorted({mask12_for_byte(b) for b in range(256)})

    def chi(m: int) -> int:
        return chi_from_q(q0, q1, m)

    # ker(chi)
    D0 = [m for m in C if chi(m) == 0]
    assert len(D0) == 64, f"expected |ker(chi)|=64, got {len(D0)}"

    # rank(D0)=6: compute rank via generator-matrix approach by spanning D0
    # build a matrix with D0 elements as rows (12-bit) and compute rank in GF(2)
    M = np.zeros((len(D0), 12), dtype=np.uint8)
    for i, m in enumerate(D0):
        for bit in range(12):
            M[i, bit] = (m >> bit) & 1
    rank_D0 = gf2_rank(M)
    assert rank_D0 == 6, f"expected rank(D0)=6, got {rank_D0}"

    # Build horizon u-sets per vertex
    U_by_v: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    for h_idx in h_idxs:
        a, _ = unpack_state(int(ont[h_idx]))
        vtx = vertex_map[a]
        u = (a ^ ARCHETYPE_A12) & 0xFFF
        U_by_v[vtx].add(u)
    assert all(len(U_by_v[v]) == 64 for v in range(4))

    # Coset check: for each vertex, pick u0 and compare u0 XOR D0 to U_v
    D0_set = set(D0)
    for v in range(4):
        u0 = next(iter(U_by_v[v]))
        coset = {(u0 ^ d) & 0xFFF for d in D0_set}
        assert coset == U_by_v[v], f"vertex {v} is not a coset of ker(chi)"

    print(SEP)
    print("H3: boundary stabilizer subgroup and vertex cosets")
    print(SEP)
    print(f"  q0=0x{q0:03x}, q1=0x{q1:03x}")
    print(f"  |ker(chi)|={len(D0)}, rank(ker(chi))={rank_D0}")
    print("  verified: each vertex u-set is exactly one coset of ker(chi)")
