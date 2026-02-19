# === secret_lab_ignore/agent/runtime_policy.py ===
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.router.kernel import RouterKernel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class PolicyConfig:
    atlas_dir: str
    model_dir: str
    policy_probs_path: str          # data/policy_probs_f16.npy
    token_unigram_path: str         # data/token_unigram.npy
    vocab_size: int
    seed: int = 0
    top_tokens_per_byte: int = 64   # resolve byte -> pick among top N tokens for that byte


@dataclass
class PolicyRunner:
    cfg: PolicyConfig

    kernel: RouterKernel = field(init=False)
    policy_probs: NDArray[np.float16] = field(init=False)   # [65536,256]
    unigram: NDArray[np.uint32] = field(init=False)         # [vocab]
    byte_top_tokens: NDArray[np.int32] = field(init=False)  # [256, topN]
    byte_top_weights: NDArray[np.float32] = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.kernel = RouterKernel(atlas_dir=Path(self.cfg.atlas_dir))
        self.policy_probs = np.load(self.cfg.policy_probs_path, mmap_mode="r")
        self.unigram = np.load(self.cfg.token_unigram_path, mmap_mode="r")
        self.rng = np.random.default_rng(self.cfg.seed)

        assert self.policy_probs.shape[0] == self.kernel.ontology.shape[0]
        assert self.policy_probs.shape[1] == 256
        assert self.unigram.shape[0] >= self.cfg.vocab_size

        self._precompute_top_tokens_per_byte()

    def _precompute_top_tokens_per_byte(self):
        vocab = self.cfg.vocab_size
        topN = self.cfg.top_tokens_per_byte

        # group tokens by last byte
        groups = [[] for _ in range(256)]
        for tid in range(vocab):
            groups[tid & 0xFF].append(tid)

        top_tokens = np.zeros((256, topN), dtype=np.int32)
        top_weights = np.zeros((256, topN), dtype=np.float32)

        uni = self.unigram[:vocab].astype(np.float64)
        uni = uni + 1.0  # avoid zeros

        for b in range(256):
            g = np.array(groups[b], dtype=np.int32)
            w = uni[g]
            idx = np.argsort(w)[::-1][:topN]
            sel = g[idx]
            ww = w[idx].astype(np.float64)
            ww = ww / (ww.sum() + 1e-18)
            top_tokens[b, :len(sel)] = sel
            top_weights[b, :len(sel)] = ww.astype(np.float32)

        self.byte_top_tokens = top_tokens
        self.byte_top_weights = top_weights

    def step_byte(self) -> int:
        s = int(self.kernel.state_index[0])
        p = self.policy_probs[s].astype(np.float32)
        # ensure sums to 1 (float16 rounding)
        p = p / (p.sum() + 1e-18)
        b = int(self.rng.choice(256, p=p))
        self.kernel.step_byte(b)
        return b

    def byte_to_token(self, b: int) -> int:
        toks = self.byte_top_tokens[b]
        w = self.byte_top_weights[b]
        # If row is partially zero-filled, renormalize nonzeros
        m = (w > 0)
        if not np.any(m):
            return int(toks[0])
        ww = w[m]
        ww = ww / (ww.sum() + 1e-18)
        sel = int(self.rng.choice(toks[m], p=ww))
        return sel
