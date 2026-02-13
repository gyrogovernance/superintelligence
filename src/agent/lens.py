from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ContextState:
    c: NDArray[np.float32]  # semantic context vector in R^(nb*nf)
    X: NDArray[np.float32]  # adaptor semantic state in R^(nb,nf)


class ContextBuilder:
    """
    Deterministic context builder replayable from token history.
    """
    def __init__(self, d: int, nb: int, nf: int, alpha: float = 0.9):
        self.d = int(d)
        self.nb = int(nb)
        self.nf = int(nf)
        self.alpha = float(alpha)

    def init(self) -> ContextState:
        return ContextState(
            c=np.zeros(self.d, dtype=np.float32),
            X=np.zeros((self.nb, self.nf), dtype=np.float32),
        )

    def update(self, st: ContextState, x_in: NDArray[np.float32]) -> None:
        assert x_in.shape == (self.d,)
        st.c = (self.alpha * st.c + (1.0 - self.alpha) * x_in).astype(np.float32)


class Lens:
    def build_O_field(self, ctx: ContextState) -> NDArray[np.float32]:
        raise NotImplementedError


@dataclass
class AdaptorLens(Lens):
    adaptor: Any

    def build_O_field(self, ctx: ContextState) -> NDArray[np.float32]:
        return self.adaptor.build_O_field_from_X(ctx.X)
