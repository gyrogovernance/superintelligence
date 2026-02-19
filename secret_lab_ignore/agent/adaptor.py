# === secret_lab_ignore/agent/adaptor.py ===
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AdaptorMeta:
    adaptor_version: str
    model_name: str
    nb: int
    nf_hidden: int
    nf_mlp: int
    K: int
    n_layers: int
    processed_layers: NDArray[np.int32]
    op_names: tuple[str, ...]
    deltas: NDArray[np.uint8]  # [8]


@dataclass(frozen=True)
class GyroAdaptor:
    meta: AdaptorMeta
    intron_sensitivity: NDArray[np.float32]   # [n_layers, 3, 8]
    intron_coherence: NDArray[np.float32]     # [n_layers, 3, 8]
    intron_pair_counts: NDArray[np.int32]     # [n_layers, 3, 8]
    vertex_flow: NDArray[np.float32]          # [n_layers, 3, 4, 4]

    @property
    def nb(self) -> int:
        return int(self.meta.nb)

    @property
    def K(self) -> int:
        return int(self.meta.K)

    # ---------------------------------------------------------------------
    # Compatibility helpers (run.py expects these)
    # ---------------------------------------------------------------------

    def layer_profiles(self) -> NDArray[np.float32]:
        """
        Per-layer mean intron sensitivity profile.
        Returns: [n_layers, 8] averaged across the 3 MLP ops.
        """
        return np.mean(self.intron_sensitivity, axis=1).astype(np.float32, copy=False)

    def layer_ranking(self) -> NDArray[np.int32]:
        """
        Layers sorted by total sensitivity (most sensitive first).
        Returns indices in 0..n_layers-1 (i.e., indices into processed_layers).
        """
        totals = np.sum(self.intron_sensitivity, axis=(1, 2)).astype(np.float64)  # [n_layers]
        return np.argsort(-totals).astype(np.int32)

    def mean_dir_weights(self) -> NDArray[np.float32]:
        """Mean direction weights [8] across layers and operators."""
        w = np.mean(self.intron_sensitivity, axis=(0, 1)).astype(np.float32, copy=False)
        if float(np.sum(w)) < 1e-10:
            w = np.ones(8, dtype=np.float32)
        return w

    def build_similarity(self, h: int) -> NDArray[np.float32]:
        """
        Similarity weights over boundary indices [256] based on weighted
        Hamming distance in byte-xor (intron) space.

        delta = h XOR h_target (byte)
        dist(delta) = sum_i w_i * bit_i(delta)
        sim = exp(-dist / sum(w))
        """
        h = int(h) & 0xFF
        w = self.mean_dir_weights()  # [8]
        wsum = float(np.sum(w)) + 1e-8

        targets = np.arange(256, dtype=np.uint16)
        delta = (targets ^ h).astype(np.uint16)

        bits = np.zeros((256, 8), dtype=np.float32)
        for i in range(8):
            bits[:, i] = ((delta >> i) & 1).astype(np.float32)

        dist = bits @ w  # [256]
        sim = np.exp(-dist / wsum).astype(np.float32)
        sim /= (float(sim.mean()) + 1e-8)
        return sim

    @classmethod
    def load(cls, path: str) -> GyroAdaptor:
        z = np.load(path, allow_pickle=False)
        version = str(z["adaptor_version"]) if "adaptor_version" in z else "1.0"

        if version.startswith("4.1"):
            meta = AdaptorMeta(
                adaptor_version=str(z["adaptor_version"]),
                model_name=str(z["model_name"]),
                nb=int(z["nb"]),
                nf_hidden=int(z["nf_hidden"]),
                nf_mlp=int(z["nf_mlp"]),
                K=int(z["K"]),
                n_layers=int(z["n_layers"]),
                processed_layers=z["processed_layers"].astype(np.int32),
                op_names=tuple(str(x) for x in z["op_names"]),
                deltas=z["deltas"].astype(np.uint8),
            )
            return cls(
                meta=meta,
                intron_sensitivity=z["intron_sensitivity"].astype(np.float32),
                intron_coherence=z["intron_coherence"].astype(np.float32),
                intron_pair_counts=z["intron_pair_counts"].astype(np.int32),
                vertex_flow=z["vertex_flow"].astype(np.float32),
            )

        raise ValueError(
            f"Adaptor version {version} not supported. "
            f"Rebuild with: python -m secret_lab_ignore.agent.adaptor_build"
        )
