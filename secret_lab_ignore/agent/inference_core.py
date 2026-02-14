from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class InferenceState:
    M: NDArray[np.float32]  # [256,4,K]
    # role-local carry
    last_h: int = 0
    last_p: int = 0
    last_chi: int = 0
    last_o: Optional[NDArray[np.float32]] = None

    @classmethod
    def create(cls, K: int) -> "InferenceState":
        return cls(M=np.zeros((256, 4, K), dtype=np.float32))


def rmsnorm(v: NDArray[np.float32], eps: float = 1e-6) -> NDArray[np.float32]:
    return v / (np.sqrt(np.mean(v * v)) + eps)


@dataclass
class InferenceRoles:
    """
    L_inf and R_inf are explicit roles to support BU Egress/Ingress structure.
    - L_inf: interpret at fixed genealogy prefix (no commit)
    - R_inf: commit-effect update at the new address after a byte is appended
    """
    K: int
    eta: float
    gamma_table: NDArray[np.float32]  # [4,4,13]
    mode: Literal["td", "hebb"] = "td"

    def retention(self, p: int, chi_prev: int, chi_curr: int, w: int) -> float:
        base = (0.995, 0.990, 0.985, 0.980)[int(p) & 3]
        if chi_prev == chi_curr:
            base *= 1.00
        elif (chi_prev ^ chi_curr) == 3:
            base *= 0.92
        else:
            base *= 0.97
        base *= (1.0 - 0.06 * (float(w) / 12.0))
        return float(np.clip(base, 0.90, 0.9995))

    def L_inf(self, sigma: InferenceState, *, h: int, chi: int, p: int, o_local: NDArray[np.float32]) -> None:
        sigma.last_h = int(h)
        sigma.last_chi = int(chi)
        sigma.last_p = int(p)
        sigma.last_o = rmsnorm(o_local.astype(np.float32, copy=False))

    def R_inf(
        self,
        sigma: InferenceState,
        *,
        b: int,
        h_next: int,
        chi_next: int,
        p_next: int,
        byte_weight: int,
        o_next: NDArray[np.float32],
    ) -> None:
        # require L_inf has run
        assert sigma.last_o is not None
        o_prev = sigma.last_o
        o_n = rmsnorm(o_next.astype(np.float32, copy=False))

        w = int(byte_weight)
        chi_prev = int(sigma.last_chi)
        chi_curr = int(chi_next)
        gamma = float(self.gamma_table[chi_prev, chi_curr, min(w, 12)])
        f = self.retention(int(p_next), chi_prev, chi_curr, w)

        hN = int(h_next)
        pN = int(p_next)

        if self.mode == "td":
            sigma.M[hN, pN, :] = (f * sigma.M[hN, pN, :] + (1.0 - f) * o_n).astype(np.float32)
        else:
            U = (gamma * (o_n * o_prev)).astype(np.float32)
            sigma.M[hN, pN, :] = (f * sigma.M[hN, pN, :] + (1.0 - f) * (self.eta * U)).astype(np.float32)

        sigma.last_o = o_n
        sigma.last_h = hN
        sigma.last_p = pN
        sigma.last_chi = chi_curr

    def Bal(self, sigma: InferenceState) -> NDArray[np.float32]:
        # balanced projection used for BU-Egress checks
        # example: per-slot energy
        X = sigma.M.reshape(-1, sigma.M.shape[-1])
        return np.sqrt(np.mean(X * X, axis=1)).astype(np.float32)


@dataclass
class InferenceEgress:
    K: int
    temperature: float
    weight_penalty: float

    features: NDArray[np.float32]     # [256,K]
    gamma_table: NDArray[np.float32]  # [4,4,13]
    byte_weight: NDArray[np.uint8]    # [256]

    def policy(
        self,
        *,
        h: int,
        chi: int,
        p: int,
        sigma: InferenceState,
        O_field: NDArray[np.float32],
        cf_h: NDArray[np.uint8],
        cf_chi: NDArray[np.uint8],
        cf_p: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        x = rmsnorm((sigma.M[int(h), int(p), :] + O_field[int(h), :]).astype(np.float32))

        scores = np.empty(256, dtype=np.float32)
        for b in range(256):
            chi1 = int(cf_chi[b])
            w = int(self.byte_weight[b])
            gamma = float(self.gamma_table[int(chi), chi1, min(w, 12)])
            scores[b] = gamma * float(self.features[b] @ x) - self.weight_penalty * float(w)

        temp = max(float(self.temperature), 1e-8)
        z = (scores - float(scores.max())) / temp
        pvec = np.exp(z).astype(np.float64)
        pvec /= (pvec.sum() + 1e-18)
        return pvec.astype(np.float32)

    def sample(self, pi: NDArray[np.float32], rng: np.random.Generator) -> int:
        return int(rng.choice(256, p=pi))
