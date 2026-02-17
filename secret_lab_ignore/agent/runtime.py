from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from src.router.kernel import RouterKernel
from .inference_core import InferenceState, InferenceRoles, InferenceEgress


@dataclass
class GyroASIConfig:
    atlas_dir: str
    K: int
    eta: float
    temperature: float = 0.7
    weight_penalty: float = 0.02
    seed: int = 0


@dataclass
class GyroASI:
    cfg: GyroASIConfig

    kernel: RouterKernel = field(init=False)
    sigma: InferenceState = field(init=False)
    roles: InferenceRoles = field(init=False)
    egress: InferenceEgress = field(init=False)

    genealogy: list[int] = field(default_factory=list)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.kernel = RouterKernel(atlas_dir=Path(self.cfg.atlas_dir))
        self.sigma = InferenceState.create(K=self.cfg.K)

        with np.load(self.cfg.atlas_dir + "/phenomenology.npz", allow_pickle=False) as phen:
            features = phen[f"features_K{self.cfg.K}"].astype(np.float32)
            byte_weight = phen["byte_weight"].astype(np.uint8)

        self.roles = InferenceRoles(K=self.cfg.K, eta=self.cfg.eta, mode="td")

        self.egress = InferenceEgress(
            K=self.cfg.K,
            temperature=self.cfg.temperature,
            weight_penalty=self.cfg.weight_penalty,
            features=features,
            byte_weight=byte_weight,
        )

        self.rng = np.random.default_rng(self.cfg.seed)

    def reset(self) -> None:
        self.kernel.reset()
        self.genealogy.clear()
        self.sigma = InferenceState.create(K=self.cfg.K)

    def step(self, x_in: NDArray[np.float32]) -> int:
        """
        One step:
        - reshape x_in into O_field[256, K]
        - L_inf at current address
        - policy sample byte
        - commit byte
        - R_inf at new address
        """
        K = self.cfg.K
        if x_in.size != 256 * K:
            raise ValueError(f"x_in must have size 256*K={256*K}, got {x_in.size}")

        O_field = x_in.reshape(256, K).astype(np.float32)

        i = int(self.kernel.state_index[0])
        last_b = int(self.kernel.last_byte[0]) & 0xFF

        h = int(self.kernel.state_horizon[i])
        chi = int(self.kernel.state_vertex[i])
        p = int(self.kernel.phase[i, last_b])

        self.roles.L_inf(self.sigma, h=h, chi=chi, p=p, o_local=O_field[h, :])

        cf_chi = self.kernel.next_vertex[i, :].astype(np.uint8, copy=False)

        pi = self.egress.policy(
            h=h, chi=chi, p=p,
            sigma=self.sigma,
            O_field=O_field,
            cf_chi=cf_chi,
        )
        b = self.egress.sample(pi, self.rng)

        self.kernel.step_byte(b)
        self.genealogy.append(b)

        i1 = int(self.kernel.state_index[0])
        h1 = int(self.kernel.state_horizon[i1])
        chi1 = int(self.kernel.state_vertex[i1])
        p1 = int(self.kernel.phase[i1, b])
        w = int(self.kernel.byte_weight[b])

        self.roles.R_inf(
            self.sigma,
            b=b,
            h_next=h1,
            chi_next=chi1,
            p_next=p1,
            byte_weight=w,
            o_next=O_field[h1, :],
        )

        return b