"""SHA-256 structural probes in chirality and WHT coordinates."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any, Sequence

import numpy as np

from src.api import try_state24_to_omega12
from src.constants import CHIRALITY_MASK_6, LAYER_MASK_12

CHIRALITY_SIZE = 64
_MASK32 = 0xFFFFFFFF
_MASK16 = 0xFFFF

SHA256_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)

SHA256_K = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)


def digest_to_chirality6_mod6_parity(digest32: bytes) -> int:
    """
    Map a 32-byte digest to one 6-bit chirality word.

    Bit i of the output is the parity of digest bits whose global bit-index
    is congruent to i modulo 6.
    """
    if len(digest32) != 32:
        raise ValueError(f"Expected 32-byte digest, got {len(digest32)} bytes")
    out = 0
    for bit_idx in range(256):
        byte_idx = bit_idx >> 3
        bit_in_byte = bit_idx & 7
        bit = (digest32[byte_idx] >> bit_in_byte) & 1
        if bit:
            out ^= 1 << (bit_idx % 6)
    return out & 0x3F


def digest_to_chirality6_words(digest32: bytes, word_count: int = 42) -> list[int]:
    """Extract non-overlapping 6-bit chirality words from a 32-byte digest."""
    if len(digest32) != 32:
        raise ValueError(f"Expected 32-byte digest, got {len(digest32)} bytes")
    if word_count <= 0 or word_count * 6 > 256:
        raise ValueError("word_count must satisfy 0 < word_count*6 <= 256")
    bits = int.from_bytes(digest32, "big")
    words: list[int] = []
    offset = 256
    for _ in range(word_count):
        offset -= 6
        words.append((bits >> offset) & CHIRALITY_MASK_6)
    return words


def chirality_words_histogram64(
    digests: Iterable[bytes],
    word_count: int = 42,
) -> np.ndarray:
    """Build a position-by-chirality histogram from digest trajectories."""
    if word_count <= 0 or word_count * 6 > 256:
        raise ValueError("word_count must satisfy 0 < word_count*6 <= 256")
    hist = np.zeros((word_count, CHIRALITY_SIZE), dtype=np.int64)
    for digest in digests:
        words = digest_to_chirality6_words(digest, word_count=word_count)
        for idx, ch in enumerate(words):
            hist[idx, ch] += 1
    return hist


def random_messages(
    sample_count: int,
    seed: int = 0,
    min_len: int = 1,
    max_len: int = 64,
) -> list[bytes]:
    """Generate deterministic pseudo-random messages with varied lengths."""
    if sample_count < 0:
        raise ValueError("sample_count must be >= 0")
    if min_len <= 0 or max_len < min_len:
        raise ValueError("Require 0 < min_len <= max_len")
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_len, max_len + 1, size=sample_count)
    return [bytes(rng.integers(0, 256, size=int(n), dtype=np.uint8)) for n in lengths]


def random_block_words(sample_count: int, seed: int = 1) -> list[tuple[int, ...]]:
    """Generate deterministic random SHA-256 16-word blocks."""
    if sample_count < 0:
        raise ValueError("sample_count must be >= 0")
    rng = np.random.default_rng(seed)
    out: list[tuple[int, ...]] = []
    for _ in range(sample_count):
        words = rng.integers(0, 1 << 32, size=16, dtype=np.uint64)
        out.append(tuple(int(w) & _MASK32 for w in words))
    return out


def sha256_block_words_from_messages(messages: Iterable[bytes], block_index: int = 0) -> list[tuple[int, ...]]:
    """Extract padded SHA-256 block words from input messages."""
    if block_index < 0:
        raise ValueError("block_index must be >= 0")
    out: list[tuple[int, ...]] = []
    for msg in messages:
        if not isinstance(msg, (bytes, bytearray)):
            raise TypeError("messages must be bytes-like")
        payload = bytes(msg)
        bit_len = len(payload) * 8
        padded = bytearray(payload)
        padded.append(0x80)
        pad_len = (56 - (len(padded) % 64)) % 64
        padded.extend(b"\x00" * pad_len)
        padded.extend(bit_len.to_bytes(8, "big"))
        if len(padded) % 64 != 0:
            raise ValueError("SHA-256 padding produced malformed output")
        block_offset = block_index * 64
        if block_offset + 64 > len(padded):
            raise ValueError("message does not have requested block index")
        block = padded[block_offset : block_offset + 64]
        out.append(tuple(int.from_bytes(block[i : i + 4], "big") for i in range(0, 64, 4)))
    return out


def sha256_digests(messages: Iterable[bytes]) -> list[bytes]:
    """Return SHA-256 digests for input messages."""
    return [hashlib.sha256(m).digest() for m in messages]


def chirality_histogram64(digests: Iterable[bytes]) -> np.ndarray:
    """Build a 64-bin chirality histogram from digest stream."""
    hist = np.zeros(CHIRALITY_SIZE, dtype=np.int64)
    for d in digests:
        chi = digest_to_chirality6_mod6_parity(d)
        hist[chi] += 1
    return hist


def _fwht_inplace(values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    n = int(arr.shape[0])
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Expected power-of-two length, got {n}")
    h = 1
    while h < n:
        step = h << 1
        for start in range(0, n, step):
            for idx in range(start, start + h):
                x = arr[idx]
                y = arr[idx + h]
                arr[idx] = x + y
                arr[idx + h] = x - y
        h = step
    arr /= float(np.sqrt(n))
    return arr


def wht64(values64: Iterable[float]) -> np.ndarray:
    """Apply canonical orthonormal 64-point fast Walsh-Hadamard transform."""
    arr = np.asarray(tuple(values64), dtype=np.float64)
    if arr.shape != (CHIRALITY_SIZE,):
        raise ValueError(f"Expected shape (64,), got {arr.shape}")
    return _fwht_inplace(arr)


def _rotr32(x: int, r: int) -> int:
    return ((x >> r) | (x << (32 - r))) & _MASK32


def _sigma0(x: int) -> int:
    return _rotr32(x, 7) ^ _rotr32(x, 18) ^ (x >> 3)


def _sigma1(x: int) -> int:
    return _rotr32(x, 17) ^ _rotr32(x, 19) ^ (x >> 10)


def _Sigma0(x: int) -> int:
    return _rotr32(x, 2) ^ _rotr32(x, 13) ^ _rotr32(x, 22)


def _Sigma1(x: int) -> int:
    return _rotr32(x, 6) ^ _rotr32(x, 11) ^ _rotr32(x, 25)


def _ch(x: int, y: int, z: int) -> int:
    return (x & y) ^ (~x & z)


def _maj(x: int, y: int, z: int) -> int:
    return (x & y) ^ (x & z) ^ (y & z)


def sha256_message_schedule(block_words: Sequence[int]) -> tuple[int, ...]:
    """Compute 64-word message schedule from one 16-word SHA-256 block."""
    if len(block_words) != 16:
        raise ValueError(f"Expected 16 input words, got {len(block_words)}")
    w = [int(x) & _MASK32 for x in block_words]
    for idx in range(16, 64):
        s0 = _sigma0(w[idx - 15])
        s1 = _sigma1(w[idx - 2])
        w.append((w[idx - 16] + s0 + w[idx - 7] + s1) & _MASK32)
    return tuple(w)


def sha256_round_chirality_trajectory(
    block_words: Sequence[int],
    initial_state: tuple[int, ...] | None = None,
) -> list[int]:
    """Track per-round 6-bit chirality on internal state (state bytes mod6 parity)."""
    if len(block_words) != 16:
        raise ValueError(f"Expected 16 input words, got {len(block_words)}")
    if initial_state is None:
        a, b, c, d, e, f, g, h = SHA256_IV
    else:
        if len(initial_state) != 8:
            raise ValueError("Expected 8 words in initial_state")
        a, b, c, d, e, f, g, h = [int(x) & _MASK32 for x in initial_state]

    schedule = sha256_message_schedule(block_words)
    trajectory: list[int] = []
    for i in range(64):
        state = (
            (a << 224)
            | (b << 192)
            | (c << 160)
            | (d << 128)
            | (e << 96)
            | (f << 64)
            | (g << 32)
            | h
        ).to_bytes(32, "big")
        trajectory.append(digest_to_chirality6_mod6_parity(state))

        s1 = _Sigma1(e)
        ch = _ch(e, f, g)
        temp1 = (h + s1 + ch + SHA256_K[i] + schedule[i]) & _MASK32
        s0 = _Sigma0(a)
        maj = _maj(a, b, c)
        temp2 = (s0 + maj) & _MASK32

        h = g
        g = f
        f = e
        e = (d + temp1) & _MASK32
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & _MASK32
    return trajectory


def schedule_depth4_frame_signatures(block_words: Sequence[int]) -> tuple[int, ...]:
    """
    Return 16 XOR signatures for depth-4 frames:
    sig_i = W[4i] xor W[4i+1] xor W[4i+2] xor W[4i+3].
    """
    sched = sha256_message_schedule(block_words)
    return tuple(
        int(
            (sched[i * 4] ^ sched[i * 4 + 1] ^ sched[i * 4 + 2] ^ sched[i * 4 + 3])
            & _MASK32
        )
        for i in range(16)
    )


def schedule_depth4_projection_lift(block_words: Sequence[int]) -> list[dict[str, int]]:
    """
    Split each 4-word schedule frame into a coarse 48-bit projection and 32-bit lift.

    The coarse projection is a deliberately collapsing 12-bit signature built from
    one 3-bit fold per frame word. The lift stores the four low-byte introns,
    mirroring the quotient/lift pattern used in holography tests.
    """
    sched = sha256_message_schedule(block_words)
    frames: list[dict[str, int]] = []
    for frame_idx in range(16):
        projection48 = 0
        lift32 = 0
        omega_mask4 = 0
        for lane, word in enumerate(sched[frame_idx * 4:frame_idx * 4 + 4]):
            a12 = (word >> 20) & LAYER_MASK_12
            b12 = (word >> 8) & LAYER_MASK_12
            intron8 = word & 0xFF
            q12 = a12 ^ b12
            q3 = (q12 ^ (q12 >> 3) ^ (q12 >> 6)) & 0x7
            projection48 |= (q3 << (lane * 3))
            lift32 = ((lift32 << 8) | intron8) & _MASK32
            state24 = (a12 << 12) | b12
            if try_state24_to_omega12(state24) is not None:
                omega_mask4 |= 1 << lane
        frames.append(
            {
                "frame": frame_idx,
                "projection48": int(projection48),
                "lift32": int(lift32),
                "omega_mask4": int(omega_mask4),
            }
        )
    return frames


def schedule_omega_trajectory(block_words: Sequence[int]) -> dict[str, Any]:
    """Map 64 schedule words to Omega trajectory probes on 12-bit paired components."""
    sched = sha256_message_schedule(block_words)
    in_omega: list[int] = []
    horizon_dist: list[int] = []
    ab_dist: list[int] = []
    intron: list[int] = []
    for word in sched:
        a12 = (word >> 20) & LAYER_MASK_12
        b12 = (word >> 8) & LAYER_MASK_12
        i12 = word & 0xFF
        state24 = (a12 << 12) | b12
        in_omega.append(1 if try_state24_to_omega12(state24) is not None else 0)
        horizon = (a12 ^ (b12 ^ LAYER_MASK_12)).bit_count()
        ab = (a12 ^ b12).bit_count()
        horizon_dist.append(horizon)
        ab_dist.append(ab)
        intron.append(i12)

    in_omega_ratio = float(sum(in_omega) / float(len(in_omega)))
    return {
        "in_omega": in_omega,
        "horizon_distance": horizon_dist,
        "ab_distance": ab_dist,
        "in_omega_ratio": in_omega_ratio,
        "horizon_mean": float(np.mean(horizon_dist)),
        "ab_mean": float(np.mean(ab_dist)),
        "horizon_hop_mean": float(np.mean(np.abs(np.diff(horizon_dist)))) if horizon_dist else 0.0,
        "ab_hop_mean": float(np.mean(np.abs(np.diff(ab_dist)))) if ab_dist else 0.0,
        "intron": intron,
    }


def _pair_diagonal_ratio(word: int) -> float:
    """
    Pairwise ratio of two-bit units in payload pair states (01 or 10).
    """
    x = int(word) & _MASK32
    cnt = 0
    for shift in range(0, 32, 2):
        pair = (x >> shift) & 0x3
        if pair in (1, 2):
            cnt += 1
    return cnt / 16.0


def _k4_sector_fractions(x: int, y: int) -> tuple[float, float, float, float]:
    x = int(x) & _MASK32
    y = int(y) & _MASK32
    x_l = x & _MASK16
    x_h = x >> 16
    y_l = y & _MASK16
    y_h = y >> 16
    d00 = x_l * y_l
    d01 = x_l * y_h
    d10 = x_h * y_l
    d11 = x_h * y_h
    total = float(d00 + (1 << 16) * (d01 + d10) + (1 << 32) * d11)
    if total <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return (
        float(d00 / total),
        float((1 << 16) * d01 / total),
        float((1 << 16) * d10 / total),
        float((1 << 32) * d11 / total),
    )


def _add_chain_carries(values: Sequence[int]) -> tuple[int, int, list[int], list[int], list[int]]:
    """
    Add a chain of u32 values left-to-right and return:
    (final, low16_carry_count, low16_carries, high16_carries, sums).
    """
    if len(values) < 2:
        raise ValueError("Need at least two values in add chain")
    acc = int(values[0]) & _MASK32
    low_carries: list[int] = []
    high_carries: list[int] = []
    sums: list[int] = []
    for nxt in values[1:]:
        right = int(nxt) & _MASK32
        low_sum = (acc & _MASK16) + (right & _MASK16)
        carry_low = 1 if low_sum > _MASK16 else 0
        high_sum = (acc >> 16) + (right >> 16) + carry_low
        carry_high = 1 if high_sum > _MASK16 else 0
        acc = (low_sum & _MASK16) | ((high_sum & _MASK16) << 16)
        low_carries.append(carry_low)
        high_carries.append(carry_high)
        sums.append(acc)
    return acc, int(sum(low_carries)), low_carries, high_carries, sums


def sha256_round_profiles(block_words: Sequence[int], initial_state: tuple[int, ...] | None = None) -> dict[str, float]:
    """
    Execute 64 rounds and return aggregate decomposition-like metrics:
    - low and high 16-bit carry fractions,
    - fraction of rounds with zero carry in both temp chains,
    - mean pair-diagonal fraction of temp1 and temp2 words.
    """
    if len(block_words) != 16:
        raise ValueError(f"Expected 16 words, got {len(block_words)}")
    if initial_state is None:
        a, b, c, d, e, f, g, h = SHA256_IV
    else:
        if len(initial_state) != 8:
            raise ValueError("Expected 8 words in initial_state")
        a, b, c, d, e, f, g, h = [int(x) & _MASK32 for x in initial_state]

    schedule = sha256_message_schedule(block_words)

    low_carry_events = 0
    high_carry_events = 0
    low_steps = 0
    high_steps = 0
    zero_chain_rounds = 0
    temp1_pair = 0.0
    temp2_pair = 0.0
    temp1_d00 = 0.0
    temp1_d01 = 0.0
    temp1_d10 = 0.0
    temp1_d11 = 0.0
    temp2_d00 = 0.0
    temp2_d01 = 0.0
    temp2_d10 = 0.0
    temp2_d11 = 0.0
    temp1_additions = 0
    temp2_additions = 0

    for i in range(64):
        s1 = _Sigma1(e)
        ch = _ch(e, f, g)
        t1_a = h
        t1_b = s1
        t1_c = ch
        t1_d = SHA256_K[i]
        t1_e = schedule[i]
        temp1 = (h + s1 + ch + SHA256_K[i] + schedule[i]) & _MASK32
        s0 = _Sigma0(a)
        maj = _maj(a, b, c)
        temp2 = (s0 + maj) & _MASK32

        _, l1, lc1, hc1, _ = _add_chain_carries((t1_a, t1_b, t1_c, t1_d, t1_e))
        _, l2, lc2, hc2, _ = _add_chain_carries((s0, maj))

        low_carry_events += l1 + l2
        high_carry_events += int(sum(hc1) + sum(hc2))
        low_steps += len(lc1) + len(lc2)
        high_steps += len(hc1) + len(hc2)
        if sum(lc1) == 0 and sum(hc1) == 0 and sum(lc2) == 0 and sum(hc2) == 0:
            zero_chain_rounds += 1

        t = t1_a
        for nxt in (t1_b, t1_c, t1_d, t1_e):
            d00, d01, d10, d11 = _k4_sector_fractions(t, nxt)
            temp1_d00 += d00
            temp1_d01 += d01
            temp1_d10 += d10
            temp1_d11 += d11
            t = (t + nxt) & _MASK32
            temp1_additions += 1

        temp1_pair += _pair_diagonal_ratio(temp1)
        temp2_pair += _pair_diagonal_ratio(temp2)
        d00, d01, d10, d11 = _k4_sector_fractions(s0, maj)
        temp2_d00 += d00
        temp2_d01 += d01
        temp2_d10 += d10
        temp2_d11 += d11
        temp2_additions += 1

        h = g
        g = f
        f = e
        e = (d + temp1) & _MASK32
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & _MASK32

    return {
        "low_carry_fraction": low_carry_events / float(low_steps),
        "high_carry_fraction": high_carry_events / float(high_steps),
        "zero_chain_round_fraction": zero_chain_rounds / 64.0,
        "temp1_pair_diag_mean": temp1_pair / 64.0,
        "temp2_pair_diag_mean": temp2_pair / 64.0,
        "temp1_k4_d00": temp1_d00 / float(temp1_additions),
        "temp1_k4_d01": temp1_d01 / float(temp1_additions),
        "temp1_k4_d10": temp1_d10 / float(temp1_additions),
        "temp1_k4_d11": temp1_d11 / float(temp1_additions),
        "temp2_k4_d00": temp2_d00 / float(temp2_additions),
        "temp2_k4_d01": temp2_d01 / float(temp2_additions),
        "temp2_k4_d10": temp2_d10 / float(temp2_additions),
        "temp2_k4_d11": temp2_d11 / float(temp2_additions),
    }


def topk_energy_fractions(coeffs64: Iterable[float], ks: tuple[int, ...]) -> dict[int, float]:
    """Return top-k spectral energy fractions from a 64-vector."""
    c = np.asarray(tuple(coeffs64), dtype=np.float64)
    if c.shape != (CHIRALITY_SIZE,):
        raise ValueError(f"Expected shape (64,), got {c.shape}")
    e = np.sort(c * c)[::-1]
    total = float(np.sum(e))
    if total <= 0.0:
        return {k: 0.0 for k in ks}
    out: dict[int, float] = {}
    for k in ks:
        kk = int(max(0, min(CHIRALITY_SIZE, k)))
        out[k] = float(np.sum(e[:kk]) / total)
    return out


def non_dc_energy_fraction(coeffs64: Iterable[float]) -> float:
    """Fraction of spectral energy outside DC (index 0)."""
    c = np.asarray(tuple(coeffs64), dtype=np.float64)
    if c.shape != (CHIRALITY_SIZE,):
        raise ValueError(f"Expected shape (64,), got {c.shape}")
    total = float(np.sum(c * c))
    if total <= 0.0:
        return 0.0
    return float((total - (c[0] * c[0])) / total)
