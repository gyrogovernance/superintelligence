from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from ..router.kernel import RouterKernel  # pyright: ignore[reportMissingImports]
from .adaptor import GyroAdaptor
from .lens import AdaptorLens, ContextBuilder, ContextState
from .inference_core import InferenceState, InferenceRoles, InferenceEgress


@dataclass
class GyroASIConfig:
    atlas_dir: str
    adaptor_path: str

    K: int
    eta: float
    temperature: float = 0.7
    weight_penalty: float = 0.02
    ctx_alpha: float = 0.9
    op_blend: float = 0.7
    drive_blend: float = 0.3
    seed: int = 0


@dataclass
class GyroASI:
    cfg: GyroASIConfig

    kernel: RouterKernel = field(init=False)
    adaptor: GyroAdaptor = field(init=False)

    ctx_builder: ContextBuilder = field(init=False)
    ctx: ContextState = field(init=False)
    lens: AdaptorLens = field(init=False)

    sigma: InferenceState = field(init=False)
    roles: InferenceRoles = field(init=False)
    egress: InferenceEgress = field(init=False)

    genealogy: list[int] = field(default_factory=list)
    token_log: list[int] = field(default_factory=list)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.kernel = RouterKernel(atlas_dir=Path(self.cfg.atlas_dir))
        self.adaptor = GyroAdaptor.load(self.cfg.adaptor_path)

        assert self.adaptor.nb == 256, "expected nb=256"
        assert self.adaptor.K == self.cfg.K, "adaptor K must match cfg.K"

        self.ctx_builder = ContextBuilder(
            d=self.adaptor.nb * self.adaptor.nf,
            nb=self.adaptor.nb,
            nf=self.adaptor.nf,
            alpha=self.cfg.ctx_alpha,
        )
        self.ctx = self.ctx_builder.init()
        self.lens = AdaptorLens(adaptor=self.adaptor)

        self.sigma = InferenceState.create(K=self.cfg.K)

        # tables from kernel phenomenology
        with np.load(self.cfg.atlas_dir + "/phenomenology.npz", allow_pickle=False) as phen:
            features = phen[f"features_K{self.cfg.K}"].astype(np.float32)
            gamma_table = phen["gamma_table"].astype(np.float32)
            byte_weight = phen["byte_weight"].astype(np.uint8)

        self.roles = InferenceRoles(K=self.cfg.K, eta=self.cfg.eta, gamma_table=gamma_table, mode="td")

        self.egress = InferenceEgress(
            K=self.cfg.K,
            temperature=self.cfg.temperature,
            weight_penalty=self.cfg.weight_penalty,
            features=features,
            gamma_table=gamma_table,
            byte_weight=byte_weight,
        )

        self.rng = np.random.default_rng(self.cfg.seed)

    def reset(self) -> None:
        self.kernel.reset()
        self.genealogy.clear()
        self.token_log.clear()
        self.sigma = InferenceState.create(K=self.cfg.K)
        self.ctx = self.ctx_builder.init()

    def step_with_semantic_vector(self, x_in: NDArray[np.float32]) -> int:
        """
        One BU microstep:
        - update deterministic context
        - build O_field
        - L_inf
        - policy sample byte
        - commit byte to kernel + genealogy
        - R_inf at new address
        """
        self.ctx_builder.update(self.ctx, x_in)

        i = int(self.kernel.state_index)
        last_b = int(self.kernel.last_byte) & 0xFF

        h = int(self.kernel.state_horizon[i])
        chi = int(self.kernel.state_vertex[i])
        p = int(self.kernel.phase[i, last_b])

        # evolve adaptor semantic state with phase-keyed compiled op family
        X_drive = self.adaptor.chart_vec(self.ctx.c)
        X_prop = self.adaptor.apply_phase_op(self.ctx.X, p)
        self.ctx.X = (
            self.cfg.op_blend * X_prop + self.cfg.drive_blend * X_drive
        ).astype(np.float32)

        O_field = self.lens.build_O_field(self.ctx)  # [256,K]

        self.roles.L_inf(self.sigma, h=h, chi=chi, p=p, o_local=O_field[h, :])

        # counterfactual arrays from phenomenology
        cf_h = self.kernel.next_horizon[i, :].astype(np.uint8, copy=False)
        cf_chi = self.kernel.next_vertex[i, :].astype(np.uint8, copy=False)
        cf_p = self.kernel.next_phase[i, :].astype(np.uint8, copy=False)

        pi = self.egress.policy(
            h=h, chi=chi, p=p,
            sigma=self.sigma,
            O_field=O_field,
            cf_h=cf_h,
            cf_chi=cf_chi,
            cf_p=cf_p,
        )
        b = self.egress.sample(pi, self.rng)

        # commit
        self.kernel.step_byte(b)
        self.genealogy.append(b)

        i1 = int(self.kernel.state_index)
        h1 = int(self.kernel.state_horizon[i1])
        chi1 = int(self.kernel.state_vertex[i1])
        p1 = int(self.kernel.phase[i1, b])

        # byte weight known from kernel table
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

    def step_with_token_id(self, token_id: int, embed_table: NDArray[np.float32]) -> int:
        """
        Replay-safe helper: consume token id, update logs, and run one microstep.
        """
        tid = int(token_id)
        self.token_log.append(tid)
        x_in = embed_table[tid].astype(np.float32)
        return self.step_with_semantic_vector(x_in)
