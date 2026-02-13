from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _check_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT size must be power of two, got {n}")


def fwht_axis(x: NDArray[np.float32], axis: int, *, normalize: bool = True) -> NDArray[np.float32]:
    """
    Fast Walsh-Hadamard transform along one axis.
    """
    y = np.swapaxes(x.astype(np.float32, copy=False), axis, -1).copy()
    n = y.shape[-1]
    _check_power_of_two(n)

    flat = y.reshape(-1, n)
    h = 1
    while h < n:
        step = h << 1
        for i in range(0, n, step):
            u = flat[:, i:i + h].copy()
            v = flat[:, i + h:i + step].copy()
            flat[:, i:i + h] = u + v
            flat[:, i + h:i + step] = u - v
        h = step

    if normalize:
        flat /= np.sqrt(float(n))
    return np.swapaxes(y, axis, -1).astype(np.float32, copy=False)


def apply_walsh_phase_op(
    X: NDArray[np.float32],
    D_phase: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Apply Walsh-diagonal phase operator.

    X: [nb, nf]
    D_phase: [nf, nf, nb]
    """
    nb, nf = X.shape
    if D_phase.shape != (nf, nf, nb):
        raise ValueError(f"D_phase shape {D_phase.shape} incompatible with X shape {X.shape}")

    X_hat = fwht_axis(X, axis=0, normalize=True)
    Y_hat = np.einsum("ock,kc->ko", D_phase.astype(np.float32, copy=False), X_hat, optimize=True)
    return fwht_axis(Y_hat.astype(np.float32), axis=0, normalize=True)


def walsh_sparse_from_operator(
    W: NDArray[np.float32],
    *,
    nb: int,
    nf: int,
    m_edges: int,
) -> tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.float32], float]:
    """
    Return sparse Walsh operator edges and capture fraction.
    """
    d = nb * nf
    if W.shape != (d, d):
        raise ValueError(f"W shape {W.shape} incompatible with nb={nb} nf={nf}")

    T = W.reshape(nb, nf, nb, nf).astype(np.float32, copy=False)
    T = fwht_axis(T, axis=0, normalize=True)
    T = fwht_axis(T, axis=2, normalize=True)

    E = np.sum(T.astype(np.float64) * T.astype(np.float64), axis=(1, 3))
    total = float(np.sum(E))

    flat = E.reshape(-1)
    m = min(int(m_edges), int(flat.shape[0]))
    idx = np.argpartition(flat, -m)[-m:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    kout = (idx // nb).astype(np.int16)
    kin = (idx % nb).astype(np.int16)
    blocks = T[kout, :, kin, :].astype(np.float32, copy=False)
    captured = float(np.sum(flat[idx]))
    frac = captured / (total + 1e-18)
    return kout, kin, blocks, frac


def apply_walsh_sparse_op(
    X: NDArray[np.float32],
    kout: NDArray[np.int16],
    kin: NDArray[np.int16],
    blocks: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Apply sparse Walsh operator in boundary Fourier domain.
    """
    nb, nf = X.shape
    X_hat = fwht_axis(X, axis=0, normalize=True)
    Y_hat = np.zeros((nb, nf), dtype=np.float32)

    V = np.einsum("eoc,ec->eo", blocks, X_hat[kin], optimize=True)
    np.add.at(Y_hat, kout, V)
    return fwht_axis(Y_hat, axis=0, normalize=True)
