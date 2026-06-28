"""Gyrocrypt — Simon + Shor on hQVM kernel. Run: python secret_lab_ignore/gyrocrypt/runner.py"""

from __future__ import annotations

from typing import Optional, Tuple


def simon(n_bits: int, secret: int) -> Optional[int]:
    from kernel.simon import simon as _simon

    return _simon(int(n_bits), int(secret))


def period(N: int, base: int, Q: int | None = None) -> Optional[int]:
    from kernel.shor import period as _period

    return _period(int(N), int(base), Q)


def factor(N: int, base: int | None = None) -> Optional[Tuple[int, int]]:
    from kernel.shor import factor as _factor

    return _factor(int(N), base)


# Deprecated alias
def orderfind(N: int, base: int, Q: int | None = None) -> Optional[int]:
    return period(N, base, Q)


__all__ = ["simon", "period", "factor", "orderfind"]
