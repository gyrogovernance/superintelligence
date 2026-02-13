from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KronFactors:
    # W ≈ Σ_r A[r] ⊗ B[r]
    A: NDArray[np.float32]  # [R, nb, nb]
    B: NDArray[np.float32]  # [R, nf, nf]
    S: NDArray[np.float32]  # [nb*nb] singular values of reshaped operator


def kron_svd(
    W_tilde: NDArray[np.float32],
    *,
    nb: int,
    nf: int,
    R: int,
) -> KronFactors:
    """
    Kronecker SVD of a charted operator.

    W_tilde is (nb*nf, nb*nf).
    Interpret indices as (boundary h in [0..nb-1], fiber c in [0..nf-1]).

    Build matrix K of shape (nb*nb, nf*nf) such that SVD(K) yields Kronecker factors.
    """
    d = nb * nf
    assert W_tilde.shape == (d, d)

    # 4-tensor: (h_out, c_out, h_in, c_in)
    T = W_tilde.reshape(nb, nf, nb, nf)

    # permute to (h_out, h_in, c_out, c_in)
    T2 = np.transpose(T, (0, 2, 1, 3))

    # flatten to K: (h_out,h_in) as rows and (c_out,c_in) as cols
    K = T2.reshape(nb * nb, nf * nf)

    U, S, Vt = np.linalg.svd(K, full_matrices=False)

    r_max = min(R, S.shape[0])
    A = np.empty((r_max, nb, nb), dtype=np.float32)
    B = np.empty((r_max, nf, nf), dtype=np.float32)

    for r in range(r_max):
        a = U[:, r].reshape(nb, nb)
        b = Vt[r, :].reshape(nf, nf)

        # distribute sqrt(sigma) symmetrically
        s = float(np.sqrt(S[r]))
        A[r] = (s * a).astype(np.float32)
        B[r] = (s * b).astype(np.float32)

    return KronFactors(A=A, B=B, S=S.astype(np.float32))


def kron_residual_energy(S: NDArray[np.float32], R: int) -> float:
    """
    Residual energy beyond rank R in the Kronecker SVD spectrum.
    """
    if R >= S.shape[0]:
        return 0.0
    tail = S[R:]
    return float(np.sum(tail * tail))


def apply_kron_rankR(
    X: NDArray[np.float32],
    factors: KronFactors,
) -> NDArray[np.float32]:
    """
    Apply Σ_r A_r X B_r^T to X.

    X is (nb, nf).
    """
    nb = factors.A.shape[1]
    nf = factors.B.shape[1]
    assert X.shape == (nb, nf)

    Y = np.zeros_like(X, dtype=np.float32)
    R = factors.A.shape[0]
    for r in range(R):
        # (nb,nb) @ (nb,nf) @ (nf,nf)^T
        Y += (factors.A[r] @ X @ factors.B[r].T).astype(np.float32)
    return Y
