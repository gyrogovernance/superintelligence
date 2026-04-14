from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DyadicVector64:
    """Dyadic exact WHT coefficients with fixed exponent for normalization."""

    numerators: tuple[int, ...]
    exponent: int


def _butterfly64(a: list[int]) -> None:
    h = 1
    while h < 64:
        for i in range(0, 64, 2 * h):
            for j in range(h):
                u = a[i + j]
                v = a[i + j + h]
                a[i + j] = u + v
                a[i + j + h] = u - v
        h *= 2


def wht64_int_forward(values: list[int]) -> list[int]:
    if len(values) != 64:
        raise ValueError("wht64_int_forward requires length 64")
    a = [int(x) for x in values]
    _butterfly64(a)
    return a


def wht64_dyadic_normalized(values: list[int]) -> DyadicVector64:
    nums = tuple(wht64_int_forward(values))
    return DyadicVector64(nums, 6)


__all__ = ["DyadicVector64", "wht64_dyadic_normalized", "wht64_int_forward"]
