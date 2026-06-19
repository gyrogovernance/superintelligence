"""
Classical audit reference — falsification only, not production.

Uses native.c scorers that build the period oracle via classical modexp
(gyroscopic_exp_mod_ladder) plus cyclic QFT / suffix DP. This path proves
what ord_N(a) *should* be when comparing against the QuBEC holonomy compiler.

Production period finding: kernel.shor → native spectral readout.
Open research compiler: kernel.holonomy → MultiCellRouter (no pow() hot path).
"""

from __future__ import annotations


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


def period_reference(N: int, base: int, Q: int | None = None) -> int | None:
    """CLASSICAL AUDIT: ord_N(base) via native C scorer (modexp coset + F_{G_X})."""
    import math

    from kernel.bindings import shor_period_u64

    nn = int(N)
    bb = int(base) % nn
    if nn <= 1 or math.gcd(bb, nn) != 1:
        return None
    B = max(1, (2 * nn.bit_length() + 5) // 6)
    q = _radix_q(int(Q) if Q is not None else _default_Q(nn, B))
    r = int(shor_period_u64(bb, nn, q))
    return r if r > 1 else None


def dlp_mag2_reference(
    N: int,
    base_g: int,
    base_h: int,
    k1: int,
    k2: int,
    Q: int | None = None,
) -> float:
    """CLASSICAL AUDIT: |ψ_{k1,k2}(1)|² via native 2D tensor/suffix scorer."""
    from kernel.bindings import dlp_2d_tensor_mag2_u64, horizon_pack_keys_u64

    nn = int(N)
    B = max(1, (2 * nn.bit_length() + 5) // 6)
    qq = _radix_q(int(Q) if Q is not None else _default_Q(nn, B))
    keys, n_cells = horizon_pack_keys_u64(nn)
    return float(
        dlp_2d_tensor_mag2_u64(
            int(base_g), int(base_h), nn, qq, int(k1), int(k2), keys, n_cells
        )
    )


__all__ = ["period_reference", "dlp_mag2_reference", "_default_Q", "_radix_q"]
