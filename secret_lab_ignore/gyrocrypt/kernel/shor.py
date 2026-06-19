"""
Production: Shor period + factor via native F_{G_X} spectral readout (native.c).

Holonomy byte compiler lives in kernel.holonomy — OPEN research, not production.
"""

from __future__ import annotations

import math


def _radix_q(raw: int) -> int:
    q = max(int(raw), 64)
    d = 1
    while 64**d < q:
        d += 1
    return 64**d


def _default_Q(N: int, B: int) -> int:
    B = max(int(B), 3)
    raw = max(int(N) * int(N), 64**B)
    return _radix_q(raw)


def period(N: int, base: int, Q: int | None = None) -> int | None:
    """ord_N(base) via native F_{G_X} readout."""
    nn = int(N)
    bb = int(base) % nn
    if nn <= 1 or math.gcd(bb, nn) != 1:
        return None
    if nn >= (1 << 63):
        raise RuntimeError(f"N={nn} needs limb shor in C (no Python path)")

    B = max(1, (2 * nn.bit_length() + 5) // 6)
    qq = _radix_q(int(Q) if Q is not None else _default_Q(nn, B))
    if qq <= 1:
        return None

    from kernel.bindings import shor_period_u64

    r = int(shor_period_u64(bb, nn, qq))
    return r if r > 1 else None


def period_report(N: int, base: int, Q: int | None = None) -> dict:
    from kernel.bindings import shor_last_path_tag

    nn = int(N)
    bb = int(base) % nn
    B = max(1, (2 * nn.bit_length() + 5) // 6)
    qq = _radix_q(int(Q) if Q is not None else _default_Q(nn, B))
    r = period(nn, bb, qq)
    return {
        "N": nn,
        "base": bb,
        "Q": qq,
        "r": r,
        "path": shor_last_path_tag(),
        "method": "native F_{G_X} spectral readout",
    }


def factor(N: int, base: int | None = None) -> tuple[int, int] | None:
    from kernel.bindings import exp_mod_ladder

    nn = int(N)
    if nn <= 2 or nn % 2 == 0:
        return None

    bb = 2 if base is None else int(base) % nn
    if math.gcd(bb, nn) != 1:
        p = math.gcd(bb, nn)
        return (p, nn // p) if 1 < p < nn else None

    r = period(nn, bb)
    if r is None or r % 2 != 0:
        return None

    x = int(exp_mod_ladder(bb, r // 2, nn))
    if x in (1, nn - 1):
        return None

    p = math.gcd(x - 1, nn)
    q = math.gcd(x + 1, nn)
    if 1 < p < nn and 1 < q < nn:
        return (min(p, q), max(p, q))
    return None


def dlp_solve(N: int, g: int, h: int) -> int | None:
    """DLP production readout not wired — fail closed."""
    _ = (N, g, h)
    return None


def dlp_mag2(
    N: int,
    base_g: int,
    base_h: int,
    k1: int,
    k2: int,
    Q: int | None = None,
) -> float:
    _ = (N, base_g, base_h, k1, k2, Q)
    return 0.0


def peak_k_for_period(Q: int, period: int, j: int = 1) -> int:
    r = int(period)
    if r <= 1:
        raise ValueError("period must be > 1")
    return int((int(j) * int(Q) + r // 2) // r)


__all__ = [
    "period",
    "period_report",
    "factor",
    "dlp_solve",
    "dlp_mag2",
    "peak_k_for_period",
    "_default_Q",
    "_radix_q",
]
