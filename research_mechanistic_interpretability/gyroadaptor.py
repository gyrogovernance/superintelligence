"""
GyroAdaptor (one-off): CGM-aware adapter on top of OLMo with TRAINED translation.

Run:
  python research_mechanistic_interpretability/gyroadaptor.py

Pipeline:
  1) Load OLMo + atlas + semantic codec
  2) Collect dataset (hidden states + kernel traces + true bytes)
  3) Train probes (hidden -> kernel labels)
  4) Train ActAdapter (hidden_last -> K=43 activation) using TRUE BYTES teacher-forced
  5) Compute Gyro targets (plan 4 bytes using ActAdapter + kernel + M, decode -> token)
  6) Train BiasHead (hidden_last -> scalar bias) to promote gyro target in candidate softmax
  7) Evaluate (rank/top-k improvements, gating stats, probe stats)

Design notes:
- No CLI flags. Edit CONFIG below.
- CPU-friendly: avoids storing logits [N,V] and avoids full-vocab delta heads.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Make src imports work
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# CONFIG (edit here)
# =============================================================================

@dataclass
class Config:
    model_path: Path = Path("data/models/Olmo-3-7B-Instruct")
    atlas_dir: Path = Path("data/atlas")
    codec_path: Path = Path("data/atlas/semantic_codec.npz")
    out_dir: Path = Path("data/gyroadaptor_out")

    device: str = "cpu"
    torch_num_threads: int = 12
    dtype: torch.dtype = torch.bfloat16

    # Collection
    max_seq_len: int = 160
    TARGET_TOKENS: int = 2048  # collect until we reach at least this many token positions
    max_prompts: int = 200     # upper bound, stop earlier if TARGET_TOKENS reached

    # Probes
    probe_layers: Tuple[int, ...] = (0, 8, 16, 24, 31)  # use a few
    probe_hidden_m: int = 512
    probe_epochs: int = 6
    probe_lr: float = 1e-3
    probe_weight_decay: float = 1e-4

    # ActAdapter (hidden_last -> activation K)
    K: int = 43
    act_hidden_m: int = 256
    act_epochs: int = 3
    act_lr: float = 2e-3
    act_weight_decay: float = 1e-4
    eta_M: float = 0.00117
    M_clip: float = 10.0

    # Byte scoring weights (match your inference defaults roughly)
    score_signal: float = 0.55
    score_weight: float = 0.25
    score_wedge: float = 0.20
    score_phase: float = 0.10
    wedge_same_bonus: float = 1.0
    wedge_adj_bonus: float = 0.2
    wedge_opp_penalty: float = -1.2
    phase_match_bonus: float = 0.25

    # Bias head (sparse control)
    cand_topk: int = 256       # candidate set size from base logits
    bias_epochs: int = 4
    bias_lr: float = 2e-3
    bias_weight_decay: float = 1e-4
    bias_max: float = 8.0      # clamp bias

    # Gating
    gap_low: float = 0.15
    gap_high: float = 0.35
    probe_mismatch_weight: float = 0.6  # how much probe mismatch amplifies gating

    # Repro
    seed: int = 7


CFG = Config()


# =============================================================================
# Atlas lens
# =============================================================================

@dataclass
class AtlasLens:
    gamma_table: np.ndarray          # [4,4,13]
    features: np.ndarray             # [256,K]
    byte_weight: np.ndarray          # [256]
    byte_charge: np.ndarray          # [256]
    phase: np.ndarray                # [N,256]
    next_phase: np.ndarray           # [N,256]
    next_horizon: np.ndarray         # [N,256]
    next_vertex: np.ndarray          # [N,256]
    state_horizon: np.ndarray        # [N]
    state_vertex: np.ndarray         # [N]
    epistemology: np.ndarray         # [N,256]

    @classmethod
    def load(cls, atlas_dir: Path, K: int) -> "AtlasLens":
        phen = atlas_dir / "phenomenology.npz"
        epi = atlas_dir / "epistemology.npy"
        with np.load(phen, allow_pickle=False) as z:
            fk = f"features_K{K}"
            if fk not in z.files:
                avail = [k for k in z.files if k.startswith("features_K")]
                raise ValueError(f"{phen} missing {fk}. Available: {avail}")
            return cls(
                gamma_table=z["gamma_table"].astype(np.float32, copy=False),
                features=z[fk].astype(np.float32, copy=False),
                byte_weight=z["byte_weight"].astype(np.uint8, copy=False),
                byte_charge=z["byte_charge"].astype(np.uint8, copy=False),
                phase=z["phase"].astype(np.uint8, copy=False),
                next_phase=z["next_phase"].astype(np.uint8, copy=False),
                next_horizon=z["next_horizon"].astype(np.uint8, copy=False),
                next_vertex=z["next_vertex"].astype(np.uint8, copy=False),
                state_horizon=z["state_horizon"].astype(np.uint8, copy=False),
                state_vertex=z["state_vertex"].astype(np.uint8, copy=False),
                epistemology=np.load(epi, mmap_mode="r"),
            )


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def popcount(x: int) -> int:
    return int(x).bit_count()


def cosine_gap(embed_tokens: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    gap = 1 - cos(embed[a], embed[b]) for token id tensors a,b shape [S]
    """
    ea = embed_tokens[a].float()
    eb = embed_tokens[b].float()
    cos = F.cosine_similarity(ea, eb, dim=-1)
    return 1.0 - cos


# =============================================================================
# Probe: hidden -> (horizon, vertex, phase)
# =============================================================================

class HorizonVertexPhaseProbe(nn.Module):
    def __init__(self, hidden_dim: int = 4096, hidden_m: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_m),
            nn.GELU(),
            nn.Linear(hidden_m, hidden_m),
            nn.GELU(),
        )
        self.out_h = nn.Linear(hidden_m, 256)
        self.out_v = nn.Linear(hidden_m, 4)
        self.out_p = nn.Linear(hidden_m, 4)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.mlp(h)
        return self.out_h(x), self.out_v(x), self.out_p(x)


# =============================================================================
# ActAdapter: hidden_last -> activation a_curr in R^K
# =============================================================================

class ActAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 4096, K: int = 43, hidden_m: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_m),
            nn.GELU(),
            nn.Linear(hidden_m, K),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.net(h_last)


# =============================================================================
# BiasHead: hidden_last -> scalar bias for gyro target token only
# =============================================================================

class BiasHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, hidden_m: int = 256, bias_max: float = 8.0) -> None:
        super().__init__()
        self.bias_max = float(bias_max)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_m),
            nn.GELU(),
            nn.Linear(hidden_m, 1),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        # positive bias helps; clamp for stability
        b = F.softplus(self.net(h_last)).squeeze(-1)
        return torch.clamp(b, 0.0, self.bias_max)


# =============================================================================
# Kernel tracing + dataset
# =============================================================================

@dataclass
class FlatDataset:
    """
    Flattened per-token examples.

    Stored tensors are CPU float16/long to be small.

    Required:
      token_ids: [N] int64
      bytes4:    [N,4] uint8
      h_after:   [N]   int64   (kernel horizon AFTER applying token bytes)
      chi_after: [N]   int64
      p_after:   [N]   int64

      step_state_idx: [N,4] int64  (state_idx BEFORE each of the 4 bytes)
      step_h:         [N,4] int64  (h BEFORE each byte)
      step_chi:       [N,4] int64
      step_p:         [N,4] int64
      step_chi_next:  [N,4] int64  (chi AFTER applying that byte)
      step_w:         [N,4] int64  (byte weight)
      step_gamma:     [N,4] float32 (gamma table lookup)
      step_byte:      [N,4] uint8   (the actual byte)

      hidden_last: [N,4096] float16
      hidden_L{L}:  [N,4096] float16 for probe layers

    gyro_target: [N] int64 (computed later)
    """
    data: Dict[str, torch.Tensor]


def build_prompts() -> List[str]:
    # Long-ish prompts so token count is meaningful.
    return [
        """The development of artificial intelligence has been transformative across research and industry.
It spans machine learning, reasoning systems, and human-computer interaction. Recent progress in language models
has raised questions about reliability, governance, and evaluation.""",
        """Quantum mechanics changed our understanding of the physical world by introducing wavefunctions,
non-commuting observables, and probabilistic measurement. A central challenge is connecting mathematical structure
to operational measurement and experimental invariants.""",
        """Mathematics spans thousands of years across human civilizations. From geometry and algebra to modern
category theory and computation, mathematical thinking provides a language for structure, proof, and design.""",
        """Economics studies how societies allocate scarce resources and coordinate decisions. Modern economies
combine markets, institutions, and public policy, creating complex feedback loops and multi-level incentives.""",
        """Governance systems require traceability, accountability, and coherent coordination. When systems scale,
the distinction between local differentiation and global coherence becomes a measurable property of the regime.""",
    ] * 50  # repeat to allow up to max_prompts
    # (You can replace this with loading a local corpus file later.)


def collect_flat_dataset(
    cfg: Config,
    model: Any,
    tok: Any,
    lens: AtlasLens,
    codec: Any,
) -> FlatDataset:
    """
    Collect a dataset without storing logits.
    """
    from src.router.kernel import RouterKernel

    device = torch.device(cfg.device)
    kernel = RouterKernel(cfg.atlas_dir)

    prompts = build_prompts()
    N_target = cfg.TARGET_TOKENS
    max_prompts = min(cfg.max_prompts, len(prompts))

    # Accumulators (python lists -> torch cat)
    acc: Dict[str, List[torch.Tensor]] = {}
    for k in (
        "token_ids", "bytes4",
        "h_after", "chi_after", "p_after",
        "step_state_idx", "step_h", "step_chi", "step_p", "step_chi_next", "step_w", "step_gamma", "step_byte",
        "hidden_last",
    ):
        acc[k] = []
    for L in cfg.probe_layers:
        acc[f"hidden_L{L}"] = []

    total = 0
    t0 = time.time()

    for pi in range(max_prompts):
        if total >= N_target:
            break

        prompt = prompts[pi]
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len)
        input_ids = enc["input_ids"].to(device)
        token_ids = input_ids[0].tolist()
        S = len(token_ids)

        with torch.inference_mode():
            out = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("hidden_states missing (need output_hidden_states=True)")

        # Pull per-token hiddens
        hidden_last = hidden_states[-1][0, :S, :].to(torch.float16).cpu()  # [S,4096]
        hidden_L: Dict[int, torch.Tensor] = {}
        for L in cfg.probe_layers:
            hidden_L[L] = hidden_states[L + 1][0, :S, :].to(torch.float16).cpu()

        # Kernel trace for this sequence (teacher-forced bytes)
        kernel.reset()
        bytes4 = np.zeros((S, 4), dtype=np.uint8)

        # per token AFTER labels
        h_after = np.zeros(S, dtype=np.int64)
        chi_after = np.zeros(S, dtype=np.int64)
        p_after = np.zeros(S, dtype=np.int64)

        # per-byte step arrays (state BEFORE each byte)
        step_state_idx = np.zeros((S, 4), dtype=np.int64)
        step_h = np.zeros((S, 4), dtype=np.int64)
        step_chi = np.zeros((S, 4), dtype=np.int64)
        step_p = np.zeros((S, 4), dtype=np.int64)
        step_chi_next = np.zeros((S, 4), dtype=np.int64)
        step_w = np.zeros((S, 4), dtype=np.int64)
        step_gamma = np.zeros((S, 4), dtype=np.float32)
        step_byte = np.zeros((S, 4), dtype=np.uint8)

        for t, tid in enumerate(token_ids):
            bs = codec.encode(int(tid))
            if len(bs) != 4:
                raise ValueError("SemanticTokenCodec must return 4 bytes")
            bytes4[t, :] = np.asarray(bs, dtype=np.uint8)

            chi_prev = int(kernel.current_vertex)

            for j, b in enumerate(bs):
                # state BEFORE stepping b
                step_state_idx[t, j] = int(kernel.state_index)
                step_h[t, j] = int(kernel.current_horizon)
                step_chi[t, j] = int(kernel.current_vertex)
                step_p[t, j] = int(kernel.current_phase)

                # compute gamma using chi_prev -> chi_next under this byte
                chi_next = int(kernel.peek_next_vertex(b))
                w = int(lens.byte_weight[int(b)])
                g = float(lens.gamma_table[chi_prev, chi_next, w])

                step_chi_next[t, j] = chi_next
                step_w[t, j] = w
                step_gamma[t, j] = g
                step_byte[t, j] = int(b)

                kernel.step_byte(b)
                chi_prev = chi_next

            # AFTER token
            h_after[t] = int(kernel.current_horizon)
            chi_after[t] = int(kernel.current_vertex)
            p_after[t] = int(kernel.current_phase)

        # Add to accumulators
        acc["token_ids"].append(torch.tensor(token_ids, dtype=torch.int64))
        acc["bytes4"].append(torch.from_numpy(bytes4.astype(np.uint8)))
        acc["h_after"].append(torch.from_numpy(h_after))
        acc["chi_after"].append(torch.from_numpy(chi_after))
        acc["p_after"].append(torch.from_numpy(p_after))

        acc["step_state_idx"].append(torch.from_numpy(step_state_idx))
        acc["step_h"].append(torch.from_numpy(step_h))
        acc["step_chi"].append(torch.from_numpy(step_chi))
        acc["step_p"].append(torch.from_numpy(step_p))
        acc["step_chi_next"].append(torch.from_numpy(step_chi_next))
        acc["step_w"].append(torch.from_numpy(step_w))
        acc["step_gamma"].append(torch.from_numpy(step_gamma))
        acc["step_byte"].append(torch.from_numpy(step_byte))

        acc["hidden_last"].append(hidden_last)
        for L in cfg.probe_layers:
            acc[f"hidden_L{L}"].append(hidden_L[L])

        total += S

        if (pi + 1) % 10 == 0 or total >= N_target:
            dt = time.time() - t0
            print(f"[collect] prompts={pi+1} total_tokens={total} elapsed={dt:.1f}s")

    # Flatten
    flat: Dict[str, torch.Tensor] = {}
    for k, parts in acc.items():
        flat[k] = torch.cat(parts, dim=0)

    # placeholder gyro_target (computed later)
    flat["gyro_target"] = torch.full((flat["token_ids"].shape[0],), -1, dtype=torch.int64)

    print(f"[collect] done: N={int(flat['token_ids'].shape[0])} token positions")
    return FlatDataset(flat)


# =============================================================================
# Scoring + M update (Torch) used in ActAdapter training and gyro target planning
# =============================================================================

class GyroScorer:
    def __init__(self, cfg: Config, lens: AtlasLens, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        self.features = torch.from_numpy(lens.features).to(device=device, dtype=torch.float32)          # [256,K]
        self.byte_weight = torch.from_numpy(lens.byte_weight.astype(np.float32) / 12.0).to(device)     # [256]
        self.byte_charge = torch.from_numpy(lens.byte_charge.astype(np.int64)).to(device)              # [256]
        self.gamma_table = torch.from_numpy(lens.gamma_table).to(device=device, dtype=torch.float32)   # [4,4,13]

        self.next_phase = lens.next_phase  # numpy [N,256]
        self.epi = lens.epistemology       # numpy [N,256]
        self.state_horizon = lens.state_horizon
        self.state_vertex = lens.state_vertex

        # M state (not differentiable)
        self.M = torch.zeros((256, 4, cfg.K), dtype=torch.float32, device=device)
        self.a_prev: Optional[torch.Tensor] = None

    def reset_memory(self) -> None:
        self.M.zero_()
        self.a_prev = None

    def update_memory(self, h: int, p: int, a_curr_detached: torch.Tensor, gamma: float) -> None:
        if self.a_prev is None:
            self.a_prev = a_curr_detached.clone()
            return
        delta = float(gamma) * (a_curr_detached * self.a_prev)
        self.M[h, p, :] += float(self.cfg.eta_M) * delta
        self.M[h, p, :].clamp_(-self.cfg.M_clip, self.cfg.M_clip)
        self.a_prev = a_curr_detached.clone()

    def score_all_bytes(self, state_idx: int, h: int, chi: int, p: int, a_curr: torch.Tensor) -> torch.Tensor:
        """
        scores: [256] (differentiable w.r.t. a_curr)
        """
        x = self.M[h, p, :] + a_curr  # [K]

        signal = torch.matmul(self.features, x)  # [256]
        weight_term = 1.0 - self.byte_weight     # [256]

        ch = self.byte_charge
        same = (ch == chi).float() * self.cfg.wedge_same_bonus
        opp = ((ch ^ chi) == 3).float() * self.cfg.wedge_opp_penalty
        adj = (1.0 - (same != 0).float() - (opp != 0).float()).clamp(0, 1) * self.cfg.wedge_adj_bonus
        wedge_term = same + opp + adj

        next_p = torch.from_numpy(self.next_phase[state_idx, :].astype(np.int64)).to(self.device)
        phase_term = (next_p == p).float() * self.cfg.phase_match_bonus

        return (
            self.cfg.score_signal * signal
            + self.cfg.score_weight * weight_term
            + self.cfg.score_wedge * wedge_term
            + self.cfg.score_phase * phase_term
        )

    def plan_bytes(self, state_idx0: int, h0: int, chi0: int, p0: int, a_curr: torch.Tensor) -> List[int]:
        """
        Plan 4 bytes by greedy argmax using current memory M.
        Note: does not mutate global M; it simulates state only.
        """
        planned: List[int] = []
        idx = int(state_idx0)
        h = int(h0)
        chi = int(chi0)
        p = int(p0)

        for _ in range(4):
            scores = self.score_all_bytes(idx, h, chi, p, a_curr)
            b = int(scores.argmax().item())
            planned.append(b)

            idx = int(self.epi[idx, b])
            h = int(self.state_horizon[idx])
            chi = int(self.state_vertex[idx])
            # phase update: by definition next_phase[idx_old, b] is phase at next state for byte b
            p = int(self.next_phase[idx, b]) if isinstance(self.next_phase, np.ndarray) else int(p)

        return planned


# =============================================================================
# Training: probes
# =============================================================================

def train_probes(cfg: Config, ds: FlatDataset) -> Dict[int, HorizonVertexPhaseProbe]:
    device = torch.device(cfg.device)
    data = ds.data
    N = int(data["token_ids"].shape[0])

    # Labels: we use AFTER-token labels as the target for token position
    y_h = data["h_after"].to(torch.int64)
    y_v = data["chi_after"].to(torch.int64)
    y_p = data["p_after"].to(torch.int64)

    # Split
    val_size = max(256, int(0.1 * N))
    train_size = N - val_size

    idx = torch.randperm(N)
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]

    probes: Dict[int, HorizonVertexPhaseProbe] = {}

    print(f"[probes] N={N} train={train_size} val={val_size}")

    for L in cfg.probe_layers:
        x = data[f"hidden_L{L}"].to(torch.float32)

        probe = HorizonVertexPhaseProbe(hidden_dim=4096, hidden_m=cfg.probe_hidden_m).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay)

        for epoch in range(cfg.probe_epochs):
            probe.train()
            # mini-batch
            bs = 64
            total_loss = 0.0
            total = 0
            for start in range(0, train_size, bs):
                bidx = idx_train[start : start + bs]
                xb = x[bidx].to(device)
                hb = y_h[bidx].to(device)
                vb = y_v[bidx].to(device)
                pb = y_p[bidx].to(device)

                lh, lv, lp = probe(xb)
                loss = F.cross_entropy(lh, hb) + F.cross_entropy(lv, vb) + F.cross_entropy(lp, pb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * int(xb.shape[0])
                total += int(xb.shape[0])

            # val
            probe.eval()
            with torch.no_grad():
                xv = x[idx_val].to(device)
                hv = y_h[idx_val].to(device)
                vv = y_v[idx_val].to(device)
                pv = y_p[idx_val].to(device)
                lh, lv, lp = probe(xv)
                ah = float((lh.argmax(-1) == hv).float().mean().item())
                av = float((lv.argmax(-1) == vv).float().mean().item())
                ap = float((lp.argmax(-1) == pv).float().mean().item())

            print(f"[probes][L{L}] epoch {epoch+1}/{cfg.probe_epochs} loss={total_loss/max(total,1):.4f} acc_h={ah:.3f} acc_v={av:.3f} acc_p={ap:.3f}")

        probes[L] = probe.cpu()

    return probes


# =============================================================================
# Training: ActAdapter (byte supervision)
# =============================================================================

def train_act_adapter(cfg: Config, ds: FlatDataset, lens: AtlasLens) -> ActAdapter:
    device = torch.device(cfg.device)
    data = ds.data
    N = int(data["token_ids"].shape[0])

    act = ActAdapter(hidden_dim=4096, K=cfg.K, hidden_m=cfg.act_hidden_m).to(device)
    opt = torch.optim.AdamW(act.parameters(), lr=cfg.act_lr, weight_decay=cfg.act_weight_decay)
    scorer = GyroScorer(cfg, lens, device=device)

    print(f"[act] training on N={N} token positions (online order), epochs={cfg.act_epochs}")

    hidden_last = data["hidden_last"].to(torch.float32)   # [N,4096]
    step_state_idx = data["step_state_idx"].to(torch.int64)   # [N,4]
    step_h = data["step_h"].to(torch.int64)
    step_chi = data["step_chi"].to(torch.int64)
    step_p = data["step_p"].to(torch.int64)
    step_chi_next = data["step_chi_next"].to(torch.int64)
    step_w = data["step_w"].to(torch.int64)
    step_gamma = data["step_gamma"].to(torch.float32)
    step_byte = data["step_byte"].to(torch.int64)  # [N,4] target bytes

    for epoch in range(cfg.act_epochs):
        act.train()
        scorer.reset_memory()

        total_loss = 0.0
        total_steps = 0

        for t in range(N):
            h_last_t = hidden_last[t].to(device)  # [4096]
            # One forward pass per token:
            a_curr = act(h_last_t)                # [K]

            # Accumulate loss over the 4 teacher-forced bytes
            token_loss = 0.0
            for j in range(4):
                idx0 = int(step_state_idx[t, j].item())
                h0 = int(step_h[t, j].item())
                chi0 = int(step_chi[t, j].item())
                p0 = int(step_p[t, j].item())

                target_b = int(step_byte[t, j].item())
                target_tensor = torch.tensor([target_b], device=device, dtype=torch.long)

                scores = scorer.score_all_bytes(idx0, h0, chi0, p0, a_curr)  # [256]
                loss_j = F.cross_entropy(scores.unsqueeze(0), target_tensor)
                token_loss = token_loss + loss_j

            opt.zero_grad()
            token_loss.backward()
            opt.step()

            total_loss += float(token_loss.item())
            total_steps += 4  # 4 byte losses per token

            # Now update M using the same a_curr (non-diff), one update per byte
            a_detached = a_curr.detach()
            for j in range(4):
                h0 = int(step_h[t, j].item())
                p0 = int(step_p[t, j].item())
                g = float(step_gamma[t, j].item())
                scorer.update_memory(h0, p0, a_detached, g)

        print(f"[act] epoch {epoch+1}/{cfg.act_epochs} avg_step_loss={total_loss / max(total_steps, 1):.4f}")

    return act.cpu()


# =============================================================================
# Compute Gyro targets using trained ActAdapter
# =============================================================================

def compute_gyro_targets(
    cfg: Config,
    ds: FlatDataset,
    lens: AtlasLens,
    codec: Any,
    embed_tokens: torch.Tensor,
    act: ActAdapter,
) -> None:
    """
    Walk dataset in order, maintaining M under teacher-forced bytes.
    At each token position t, compute gyro target by planning 4 bytes via ActAdapter+M from the
    current state BEFORE consuming the true token bytes.

    Writes ds.data["gyro_target"] in-place.
    """
    device = torch.device(cfg.device)
    data = ds.data
    N = int(data["token_ids"].shape[0])

    scorer = GyroScorer(cfg, lens, device=device)
    scorer.reset_memory()
    act = act.to(device).eval()

    hidden_last = data["hidden_last"].to(torch.float32)
    token_ids = data["token_ids"].to(torch.int64)
    bytes4 = data["bytes4"].to(torch.int64)  # [N,4]

    # We need the per-token "state before first byte". That's step_state_idx[:,0], step_h[:,0], etc.
    step_state_idx = data["step_state_idx"].to(torch.int64)
    step_h = data["step_h"].to(torch.int64)
    step_chi = data["step_chi"].to(torch.int64)
    step_p = data["step_p"].to(torch.int64)
    step_gamma = data["step_gamma"].to(torch.float32)

    gyro = torch.empty((N,), dtype=torch.int64)

    print(f"[gyro_target] computing gyro targets for N={N}...")
    t0 = time.time()

    for t in range(N):
        # state BEFORE consuming true token bytes
        idx0 = int(step_state_idx[t, 0].item())
        h0 = int(step_h[t, 0].item())
        chi0 = int(step_chi[t, 0].item())
        p0 = int(step_p[t, 0].item())

        a_curr = act(hidden_last[t].to(device))

        planned_bytes = scorer.plan_bytes(idx0, h0, chi0, p0, a_curr)

        # decode using probe vector = hidden_last[t] (torch) and real embedding table
        probe = hidden_last[t].float()
        gyro_tid = int(codec.decode(planned_bytes, probe, embed_tokens))  # type: ignore[attr-defined]
        gyro[t] = gyro_tid

        # now update memory using teacher forced true bytes (not planned)
        # Use the gamma stored for each byte-step and update at step coords
        # (we use step_h/p at each byte; already in dataset arrays)
        for j in range(4):
            h_step = int(step_h[t, j].item())
            p_step = int(step_p[t, j].item())
            g = float(step_gamma[t, j].item())
            scorer.update_memory(h_step, p_step, a_curr.detach(), g)

        if (t + 1) % 500 == 0 or (t + 1) == N:
            dt = time.time() - t0
            print(f"  {t+1}/{N} done ({dt:.1f}s)")

    data["gyro_target"] = gyro
    # diagnostics
    same = float((gyro == token_ids).float().mean().item())
    print(f"[gyro_target] done. gyro==true_token rate: {same:.3f}")


# =============================================================================
# Train BiasHead: promote gyro target in candidate softmax
# =============================================================================

def train_bias_head(
    cfg: Config,
    ds: FlatDataset,
    model: Any,
) -> BiasHead:
    """
    Training objective (candidate softmax):
      - candidates = topK base logits + gyro_target
      - add bias to gyro_target logit
      - CE loss on candidate set (target=gyro_target)
    """
    device = torch.device(cfg.device)
    data = ds.data
    N = int(data["token_ids"].shape[0])

    hidden_last = data["hidden_last"].to(torch.float32)  # [N,4096]
    gyro_target = data["gyro_target"].to(torch.int64)    # [N]

    # We need lm_head weights; we avoid full forward.
    lm_head = model.lm_head
    lm_head = lm_head.to(device).eval()

    head = BiasHead(hidden_dim=4096, hidden_m=256, bias_max=cfg.bias_max).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.bias_lr, weight_decay=cfg.bias_weight_decay)

    # Split
    val_size = max(256, int(0.1 * N))
    train_size = N - val_size
    perm = torch.randperm(N)
    idx_train = perm[:train_size]
    idx_val = perm[train_size:]

    print(f"[bias] N={N} train={train_size} val={val_size} cand_topk={cfg.cand_topk}")

    bs = 8
    for epoch in range(cfg.bias_epochs):
        head.train()
        total_loss = 0.0
        total = 0

        for start in range(0, train_size, bs):
            bidx = idx_train[start : start + bs]
            h = hidden_last[bidx].to(device)           # [B,4096]
            g = gyro_target[bidx].to(device)           # [B]

            with torch.no_grad():
                base_logits = lm_head(h.to(dtype=lm_head.weight.dtype))  # [B,V]
                topv, topi = base_logits.topk(cfg.cand_topk, dim=-1)     # [B,K]

            bias = head(h)  # [B]

            # build candidate sets per row (small loop B<=8)
            losses: List[torch.Tensor] = []
            for i in range(h.shape[0]):
                cand_idx = topi[i].tolist()
                gt = int(g[i].item())
                if gt not in cand_idx:
                    cand_idx.append(gt)
                cand_idx_t = torch.tensor(cand_idx, device=device, dtype=torch.int64)
                cand_logits = base_logits[i].index_select(0, cand_idx_t)  # [C]

                # add bias to gt position
                gt_pos = cand_idx.index(gt)
                cand_logits = cand_logits.clone()
                cand_logits[gt_pos] = cand_logits[gt_pos] + bias[i]

                # CE on candidate set
                loss_i = F.cross_entropy(cand_logits.unsqueeze(0), torch.tensor([gt_pos], device=device))
                # regularize bias so it doesn't explode
                loss_i = loss_i + 0.001 * (bias[i] ** 2)
                losses.append(loss_i)

            loss = torch.stack(losses).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(h.shape[0])
            total += int(h.shape[0])

        # validation: report gyro rank/topk on candidate set and approx full rank on base
        head.eval()
        with torch.no_grad():
            # sample val subset for speed
            val_take = idx_val[: min(len(idx_val), 512)]
            h = hidden_last[val_take].to(device)
            g = gyro_target[val_take].to(device)

            base_logits = lm_head(h.to(dtype=lm_head.weight.dtype))
            base_top1 = base_logits.argmax(dim=-1)
            base_top5 = base_logits.topk(5, dim=-1).indices

            bias = head(h)
            # apply sparse bias to gyro_target logit for evaluation
            boosted = base_logits.clone()
            boosted[torch.arange(boosted.shape[0], device=device), g] += bias

            boost_top1 = boosted.argmax(dim=-1)
            boost_top5 = boosted.topk(5, dim=-1).indices

            base_hit1 = float((base_top1 == g).float().mean().item())
            base_hit5 = float((base_top5 == g.unsqueeze(-1)).any(dim=-1).float().mean().item())
            boost_hit1 = float((boost_top1 == g).float().mean().item())
            boost_hit5 = float((boost_top5 == g.unsqueeze(-1)).any(dim=-1).float().mean().item())

        print(
            f"[bias] epoch {epoch+1}/{cfg.bias_epochs} loss={total_loss/max(total,1):.4f} "
            f"hit1 {base_hit1:.3f}->{boost_hit1:.3f}  hit5 {base_hit5:.3f}->{boost_hit5:.3f}"
        )

    return head.cpu()


# =============================================================================
# Evaluation with gating (uses probe mismatch + gap)
# =============================================================================

def evaluate(
    cfg: Config,
    model: Any,
    tok: Any,
    lens: AtlasLens,
    codec: Any,
    embed_tokens: torch.Tensor,
    act: ActAdapter,
    bias_head: BiasHead,
    probes: Dict[int, HorizonVertexPhaseProbe],
) -> None:
    from src.router.kernel import RouterKernel

    device = torch.device(cfg.device)
    kernel = RouterKernel(cfg.atlas_dir)

    act = act.to(device).eval()
    bias_head = bias_head.to(device).eval()

    # pick one probe layer for gating mismatch (use L16 if available else last)
    gate_L = 16 if 16 in probes else sorted(probes.keys())[-1]
    probe = probes[gate_L].to(device).eval()

    prompts = [
        "The future of computing depends on",
        "Scientific progress requires",
        "Governance systems must ensure",
    ]

    lm_head = model.lm_head.to(device).eval()

    print("\n" + "=" * 70)
    print("[EVAL] teacher-forced evaluation on prompts")
    print("=" * 70)

    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len)
        input_ids = enc["input_ids"].to(device)
        token_ids = input_ids[0].tolist()
        S = len(token_ids)

        with torch.inference_mode():
            out = model(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("hidden_states missing")
        hidden_last = hs[-1][0, :S, :].to(torch.float32)
        hidden_gate = hs[gate_L + 1][0, :S, :].to(torch.float32)

        # compute base logits once
        with torch.no_grad():
            base_logits = lm_head(hidden_last.to(dtype=lm_head.weight.dtype)).float()  # [S,V]
        base_top1 = base_logits.argmax(dim=-1)
        base_top5 = base_logits.topk(5, dim=-1).indices

        # kernel walk + M memory
        scorer = GyroScorer(cfg, lens, device=device)
        scorer.reset_memory()
        kernel.reset()

        gyro_targets = torch.empty((S,), dtype=torch.int64, device=device)

        # collect kernel AFTER-token labels for probe mismatch
        h_after = torch.empty((S,), dtype=torch.int64, device=device)
        chi_after = torch.empty((S,), dtype=torch.int64, device=device)
        p_after = torch.empty((S,), dtype=torch.int64, device=device)

        # precompute bytes for teacher forcing
        bytes4 = [codec.encode(int(tid)) for tid in token_ids]

        for t in range(S):
            # state BEFORE token bytes
            idx0 = int(kernel.state_index)
            h0 = int(kernel.current_horizon)
            chi0 = int(kernel.current_vertex)
            p0 = int(kernel.current_phase)

            a_curr = act(hidden_last[t].to(device))

            planned = scorer.plan_bytes(idx0, h0, chi0, p0, a_curr)
            gyro_tid = int(codec.decode(planned, hidden_last[t].cpu(), embed_tokens))  # decode uses CPU tensors OK
            gyro_targets[t] = gyro_tid

            # now teacher force true bytes and update M
            chi_prev = int(kernel.current_vertex)
            for j, b in enumerate(bytes4[t]):
                chi_next = int(kernel.peek_next_vertex(b))
                w = int(lens.byte_weight[int(b)])
                g = float(lens.gamma_table[chi_prev, chi_next, w])
                scorer.update_memory(int(kernel.current_horizon), int(kernel.current_phase), a_curr.detach(), g)
                kernel.step_byte(b)
                chi_prev = chi_next

            h_after[t] = int(kernel.current_horizon)
            chi_after[t] = int(kernel.current_vertex)
            p_after[t] = int(kernel.current_phase)

        # compute gap and probe mismatch
        gap = cosine_gap(embed_tokens, base_top1.cpu(), gyro_targets.cpu()).to(device)

        with torch.no_grad():
            lh, lv, lp = probe(hidden_gate.to(device))
            ph = F.softmax(lh, dim=-1)
            pv = F.softmax(lv, dim=-1)
            pp = F.softmax(lp, dim=-1)

            # mismatch = 1 - prob(true_label)
            mh = 1.0 - ph[torch.arange(S, device=device), h_after]
            mv = 1.0 - pv[torch.arange(S, device=device), chi_after]
            mp = 1.0 - pp[torch.arange(S, device=device), p_after]
            mismatch = (mh + mv + mp) / 3.0

        # gating scalar in {0, 0.5, 1} based on (gap + w*mismatch)
        eff = gap + cfg.probe_mismatch_weight * mismatch
        gate = torch.zeros_like(eff)
        gate[eff > cfg.gap_high] = 1.0
        gate[(eff > cfg.gap_low) & (eff <= cfg.gap_high)] = 0.5

        # apply sparse bias
        with torch.no_grad():
            bias = bias_head(hidden_last.to(device))  # [S]
            boosted = base_logits.clone()
            boosted[torch.arange(S, device=device), gyro_targets] += gate * bias

            boost_top1 = boosted.argmax(dim=-1)
            boost_top5 = boosted.topk(5, dim=-1).indices

            base_hit1 = float((base_top1 == gyro_targets).float().mean().item())
            base_hit5 = float((base_top5 == gyro_targets.unsqueeze(-1)).any(dim=-1).float().mean().item())
            boost_hit1 = float((boost_top1 == gyro_targets).float().mean().item())
            boost_hit5 = float((boost_top5 == gyro_targets.unsqueeze(-1)).any(dim=-1).float().mean().item())

            intervention = float((gate > 0).float().mean().item())
            avg_gap = float(gap.mean().item())
            avg_mis = float(mismatch.mean().item())

        print("\n" + "-" * 70)
        print(f"Prompt: {prompt}")
        print(f"  base hit@1={base_hit1:.3f}  hit@5={base_hit5:.3f}")
        print(f"  boosted hit@1={boost_hit1:.3f}  hit@5={boost_hit5:.3f}")
        print(f"  avg gap={avg_gap:.3f}  avg mismatch={avg_mis:.3f}  intervention rate={intervention:.3f}")

        # Optional: show teacher-forced decoded argmax sequences (rough)
        base_text = tok.decode(base_top1.tolist(), skip_special_tokens=False)
        boost_text = tok.decode(boost_top1.tolist(), skip_special_tokens=False)
        print("\n  Teacher-forced decode (base argmax):")
        print("  " + base_text[:240].replace("\n", " "))
        print("\n  Teacher-forced decode (boost argmax):")
        print("  " + boost_text[:240].replace("\n", " "))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_seed(CFG.seed)
    CFG.out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(CFG.torch_num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(CFG.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(CFG.torch_num_threads))

    print("=" * 70)
    print("GyroAdaptor ONE-OFF (improved): translator + gyro targets + sparse bias head")
    print("=" * 70)
    print(f"device={CFG.device}  dtype={CFG.dtype}  threads={CFG.torch_num_threads}")
    print(f"TARGET_TOKENS={CFG.TARGET_TOKENS} max_seq_len={CFG.max_seq_len}")
    print(f"probe_layers={CFG.probe_layers}  K={CFG.K}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.agent.adapters import SemanticTokenCodec

    device = torch.device(CFG.device)

    print("\n[load] model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(CFG.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_path,
        dtype=CFG.dtype,
        device_map={"": CFG.device},
    )
    model.eval()

    embed_tokens = model.model.embed_tokens.weight.detach().float().cpu()
    codec = SemanticTokenCodec.load(CFG.codec_path)

    print("\n[load] atlas lens...")
    lens = AtlasLens.load(CFG.atlas_dir, K=CFG.K)

    # 1) Collect
    print("\n" + "=" * 70)
    print("[1] COLLECT DATASET")
    print("=" * 70)
    ds = collect_flat_dataset(CFG, model, tok, lens, codec)

    # 2) Train probes
    print("\n" + "=" * 70)
    print("[2] TRAIN PROBES")
    print("=" * 70)
    probes = train_probes(CFG, ds)

    # 3) Train ActAdapter
    print("\n" + "=" * 70)
    print("[3] TRAIN ACT ADAPTER (byte supervision)")
    print("=" * 70)
    act = train_act_adapter(CFG, ds, lens)

    # 4) Compute gyro targets
    print("\n" + "=" * 70)
    print("[4] COMPUTE GYRO TARGETS (planning + decode)")
    print("=" * 70)
    compute_gyro_targets(CFG, ds, lens, codec, embed_tokens, act)

    # 5) Train BiasHead
    print("\n" + "=" * 70)
    print("[5] TRAIN BIAS HEAD (sparse control)")
    print("=" * 70)
    bias_head = train_bias_head(CFG, ds, model)

    # Save artifacts
    print("\n" + "=" * 70)
    print("[save] writing artifacts")
    print("=" * 70)
    torch.save(act.state_dict(), CFG.out_dir / "act_adapter.pt")
    torch.save(bias_head.state_dict(), CFG.out_dir / "bias_head.pt")
    for L, p in probes.items():
        torch.save(p.state_dict(), CFG.out_dir / f"probe_L{L}.pt")
    torch.save(ds.data, CFG.out_dir / "dataset.pt")
    print(f"saved to {CFG.out_dir}")

    # 6) Evaluate
    print("\n" + "=" * 70)
    print("[6] EVAL")
    print("=" * 70)
    evaluate(CFG, model, tok, lens, codec, embed_tokens, act, bias_head, probes)


if __name__ == "__main__":
    main()