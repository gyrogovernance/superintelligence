from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .kron import KronFactors


@dataclass(frozen=True)
class AdaptorMeta:
    adaptor_version: str
    model_name: str
    nb: int
    nf: int
    R: int
    operators: tuple[str, ...]
    residual_energy: dict[str, float]
    tail_fraction: dict[str, float]
    operator_set_hash: str
    build_timestamp_utc: str
    build_status: str
    orthogonality_error: float


@dataclass(frozen=True)
class GyroAdaptor:
    """
    External-manifold adaptor, weights-only.

    Provides:
    - boundary chart U (nb x nb)
    - compiled operators W ~= sum_r A_r kron B_r (stored as KronFactors)
    - deterministic way to produce O_field[nb,K] from semantic vector x in R^(nb*nf)
    """
    meta: AdaptorMeta
    U: NDArray[np.float32]  # [nb, nb] orthogonal
    boundary_basis: str
    # operator name -> KronFactors
    ops: dict[str, KronFactors]
    # phase-indexed lookup directions [p, nb, nf]
    D_phase: Optional[NDArray[np.float32]]
    # reducer from fiber nf -> K (defaults to identity if K==nf)
    R_fiber: NDArray[np.float32]  # [K, nf]

    @property
    def nb(self) -> int:
        return int(self.meta.nb)

    @property
    def nf(self) -> int:
        return int(self.meta.nf)

    @property
    def K(self) -> int:
        return int(self.R_fiber.shape[0])

    def chart_vec(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        x in R^(nb*nf) -> X in R^(nb,nf) with boundary chart applied.

        X = reshape( (U ⊗ I_nf) x )
        Implementation: reshape x to (nb,nf), then multiply boundary axis by U.
        """
        nb, nf = self.nb, self.nf
        assert x.shape == (nb * nf,)
        X0 = x.reshape(nb, nf)
        X = (self.U @ X0).astype(np.float32)
        return X

    def unchart_vec(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        nb, nf = self.nb, self.nf
        assert X.shape == (nb, nf)
        X0 = (self.U.T @ X).astype(np.float32)
        return X0.reshape(nb * nf)

    def build_O_field_from_x(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Produce O_field[nb,K] from semantic vector x by:
        - charting into X[nb,nf]
        - reducing fiber: O[h] = R_fiber @ X[h]
        """
        X = self.chart_vec(x)  # [nb,nf]
        O = (X @ self.R_fiber.T).astype(np.float32)  # [nb,K]
        return O

    def build_O_field_from_X(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Produce O_field[nb,K] directly from adaptor semantic state X[nb,nf].
        """
        nb, nf = self.nb, self.nf
        assert X.shape == (nb, nf)
        return (X @ self.R_fiber.T).astype(np.float32)

    def apply_phase_op(self, X: NDArray[np.float32], phase: int) -> NDArray[np.float32]:
        """
        Read from parameter manifold at the given phase.
        If phase directions are unavailable, return X unchanged.
        """
        nb, nf = self.nb, self.nf
        assert X.shape == (nb, nf)
        p = int(phase) & 3

        if self.D_phase is not None:
            D = self.D_phase[p]
            if D.shape != (nb, nf):
                raise ValueError(f"D_phase[{p}] shape {D.shape} incompatible with X shape {X.shape}")
            proj = np.sum(X * D, axis=1, keepdims=True).astype(np.float32)
            return (proj * D).astype(np.float32)

        return X.astype(np.float32, copy=True)

    @classmethod
    def load(cls, path: str) -> "GyroAdaptor":
        z = np.load(path, allow_pickle=False)
        operators = tuple(str(x) for x in z["operators"])
        meta = AdaptorMeta(
            adaptor_version=str(z["adaptor_version"]) if "adaptor_version" in z else "1.0",
            model_name=str(z["model_name"]),
            nb=int(z["nb"]),
            nf=int(z["nf"]),
            R=int(z["R"]),
            operators=operators,
            residual_energy={k: float(z[f"resid_{k}"]) for k in operators if f"resid_{k}" in z},
            tail_fraction={k: float(z[f"tail_frac_{k}"]) for k in operators if f"tail_frac_{k}" in z},
            operator_set_hash=str(z["operator_set_hash"]) if "operator_set_hash" in z else "",
            build_timestamp_utc=str(z["build_timestamp_utc"]) if "build_timestamp_utc" in z else "",
            build_status=str(z["build_status"]) if "build_status" in z else "unknown",
            orthogonality_error=float(z["orthogonality_error"]) if "orthogonality_error" in z else float("nan"),
        )
        U = z["U"].astype(np.float32)
        basis = str(z["boundary_basis"]) if "boundary_basis" in z else "chart"
        R_fiber = z["R_fiber"].astype(np.float32)
        ops: dict[str, KronFactors] = {}
        for name in meta.operators:
            a_key = f"A_{name}"
            b_key = f"B_{name}"
            s_key = f"S_{name}"
            if a_key in z and b_key in z and s_key in z:
                A = z[a_key].astype(np.float32)
                B = z[b_key].astype(np.float32)
                S = z[s_key].astype(np.float32)
                ops[name] = KronFactors(A=A, B=B, S=S)

        D_phase = None
        if "D_phase" in z:
            D_phase = z["D_phase"].astype(np.float32)
            if D_phase.ndim != 3 or D_phase.shape[0] != 4:
                raise ValueError("D_phase has unexpected shape; rebuild adaptor with lookup directions.")

        return cls(
            meta=meta,
            U=U,
            boundary_basis=basis,
            ops=ops,
            D_phase=D_phase,
            R_fiber=R_fiber,
        )
