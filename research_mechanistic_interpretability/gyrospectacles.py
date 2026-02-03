"""
GyroSpectacles v10 (Adapter-driven): Use TRAINED GyroAdaptor artifacts to steer OLMo.

Run:
  python research_mechanistic_interpretability/gyrospectacles.py
"""

from __future__ import annotations

import os
import sys
import time
import gc
import traceback
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure "src" imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# Config - using regular class to avoid dataclass issues
# =============================================================================

class SpectaclesConfig:
    """Configuration for GyroSpectacles."""
    
    def __init__(self) -> None:
        # Paths
        self.model_path = Path("data/models/Olmo-3-7B-Instruct")
        self.atlas_dir = Path("data/atlas")
        self.codec_path = Path("data/atlas/semantic_codec.npz")
        self.adaptor_dir = Path("data/gyroadaptor_out")
        self.act_adapter_path = Path("data/gyroadaptor_out/act_adapter.pt")
        self.bias_head_path = Path("data/gyroadaptor_out/bias_head.pt")
        self.probe_path = Path("data/gyroadaptor_out/probe_L16.pt")
        
        # Probe layer
        self.probe_layer_for_gate: int = 16
        
        # Dimensions
        self.hidden_dim: int = 4096
        self.K: int = 43
        
        # Runtime
        self.max_prompt_tokens: int = 160
        self.torch_num_threads: int = 12
        self.device: str = "cpu"
        self.dtype = torch.bfloat16
        
        # Bias control
        self.bias_max: float = 8.0
        
        # Gating thresholds
        self.gap_low: float = 0.15
        self.gap_high: float = 0.35
        self.probe_mismatch_weight: float = 0.6
        
        # Byte scoring weights
        self.score_signal: float = 0.55
        self.score_weight: float = 0.25
        self.score_wedge: float = 0.20
        self.score_phase: float = 0.10
        
        # Wedge bonuses/penalties
        self.wedge_same_bonus: float = 1.0
        self.wedge_adj_bonus: float = 0.2
        self.wedge_opp_penalty: float = -1.2
        
        # Phase matching
        self.phase_match_bonus: float = 0.25
        
        # Reporting
        self.teacher_forced_max_tokens: int = 120


CFG = SpectaclesConfig()

TEST_PROMPTS: Tuple[str, ...] = (
    """The development of artificial intelligence has been transformative.
From academic research to everyday applications, AI has changed how we interact
with technology. Machine learning revolutionized the field in the 2010s.""",
    """Quantum mechanics changed our understanding of the physical world.
The theory emerged to explain phenomena classical physics could not account for,
such as blackbody radiation and the photoelectric effect.""",
    """Mathematics spans thousands of years across human civilizations.
From earliest counting systems to modern abstract structures, mathematical
thought represents humanity's greatest intellectual achievements.""",
)


# =============================================================================
# Atlas lens
# =============================================================================

class AtlasLens(NamedTuple):
    gamma_table: np.ndarray
    features: np.ndarray
    byte_weight: np.ndarray
    byte_charge: np.ndarray
    next_phase: np.ndarray
    state_horizon: np.ndarray
    state_vertex: np.ndarray
    epistemology: np.ndarray


def load_atlas_lens(atlas_dir: Path, K: int) -> AtlasLens:
    phen = atlas_dir / "phenomenology.npz"
    epi = atlas_dir / "epistemology.npy"
    with np.load(phen, allow_pickle=False) as z:
        fk = f"features_K{K}"
        if fk not in z.files:
            avail = [k for k in z.files if k.startswith("features_K")]
            raise ValueError(f"{phen} missing {fk}. Available: {avail}")
        return AtlasLens(
            gamma_table=z["gamma_table"].astype(np.float32, copy=False),
            features=z[fk].astype(np.float32, copy=False),
            byte_weight=z["byte_weight"].astype(np.uint8, copy=False),
            byte_charge=z["byte_charge"].astype(np.uint8, copy=False),
            next_phase=z["next_phase"].astype(np.uint8, copy=False),
            state_horizon=z["state_horizon"].astype(np.uint8, copy=False),
            state_vertex=z["state_vertex"].astype(np.uint8, copy=False),
            epistemology=np.load(epi, mmap_mode="r"),
        )


# =============================================================================
# Trained modules
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
        b = F.softplus(self.net(h_last)).squeeze(-1)
        return torch.clamp(b, 0.0, self.bias_max)


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
# Memory + scoring
# =============================================================================

class MemoryColumn:
    def __init__(self, K: int, eta: float = 0.00117, m_clip: float = 10.0, device: Optional[torch.device] = None) -> None:
        self.K = K
        self.eta = float(eta)
        self.m_clip = float(m_clip)
        self.device = device or torch.device("cpu")
        self.M = torch.zeros((256, 4, K), dtype=torch.float32, device=self.device)
        self.a_prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.M.zero_()
        self.a_prev = None

    def update(self, h: int, p: int, a_curr: torch.Tensor, gamma: float) -> None:
        a = a_curr.to(dtype=torch.float32)
        if self.a_prev is None:
            self.a_prev = a.detach().clone()
            return
        self.M[h, p, :] += self.eta * float(gamma) * (a.detach() * self.a_prev)
        self.M[h, p, :].clamp_(-self.m_clip, self.m_clip)
        self.a_prev = a.detach().clone()


class GyroPlanner:
    def __init__(self, cfg: SpectaclesConfig, lens: AtlasLens, embed_tokens: torch.Tensor, codec: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.lens = lens
        self.codec = codec
        self.device = device

        self.embed_tokens = embed_tokens.float().cpu()
        self.features = torch.from_numpy(lens.features).to(device=device, dtype=torch.float32)
        self.byte_weight = torch.from_numpy(lens.byte_weight.astype(np.float32) / 12.0).to(device)
        self.byte_charge = torch.from_numpy(lens.byte_charge.astype(np.int64)).to(device)
        self.next_phase = lens.next_phase
        self.epi = lens.epistemology
        self.state_horizon = lens.state_horizon
        self.state_vertex = lens.state_vertex

        self.memory = MemoryColumn(K=cfg.K, eta=0.00117, m_clip=10.0, device=device)

    def reset(self) -> None:
        self.memory.reset()

    def score_all_bytes(self, idx: int, h: int, chi: int, p: int, a_curr: torch.Tensor) -> torch.Tensor:
        x = self.memory.M[h, p, :] + a_curr.to(dtype=torch.float32)
        signal = torch.matmul(self.features, x)
        weight_term = 1.0 - self.byte_weight

        ch = self.byte_charge
        same = (ch == chi).float() * self.cfg.wedge_same_bonus
        opp = ((ch ^ chi) == 3).float() * self.cfg.wedge_opp_penalty
        adj = (1.0 - (same != 0).float() - (opp != 0).float()).clamp(0, 1) * self.cfg.wedge_adj_bonus
        wedge_term = same + opp + adj

        next_p = torch.from_numpy(self.next_phase[idx, :].astype(np.int64)).to(self.device)
        phase_term = (next_p == p).float() * self.cfg.phase_match_bonus

        return (
            self.cfg.score_signal * signal
            + self.cfg.score_weight * weight_term
            + self.cfg.score_wedge * wedge_term
            + self.cfg.score_phase * phase_term
        )

    def plan_bytes(self, idx0: int, h0: int, chi0: int, p0: int, a_curr: torch.Tensor) -> List[int]:
        planned: List[int] = []
        idx, h, chi, p = int(idx0), int(h0), int(chi0), int(p0)

        for _ in range(4):
            scores = self.score_all_bytes(idx, h, chi, p, a_curr)
            b = int(scores.argmax().item())
            planned.append(b)

            idx_next = int(self.epi[idx, b])
            h_next = int(self.state_horizon[idx_next])
            chi_next = int(self.state_vertex[idx_next])
            p_next = int(self.next_phase[idx, b])

            idx, h, chi, p = idx_next, h_next, chi_next, p_next

        return planned

    def decode_token(self, planned_bytes: List[int], probe: torch.Tensor) -> int:
        return int(self.codec.decode(planned_bytes, probe.float().cpu(), self.embed_tokens))


# =============================================================================
# Metrics helpers
# =============================================================================

def nll_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    target = input_ids[:, 1:].contiguous()
    pred = logits[:, :-1, :].float().contiguous()
    logp = F.log_softmax(pred, dim=-1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(nll.mean().item())


def kl_div_fn(base_logits: torch.Tensor, new_logits: torch.Tensor) -> float:
    p = base_logits.float()
    q = new_logits.float()
    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    p_prob = p_log.exp()
    return float((p_prob * (p_log - q_log)).sum(dim=-1).mean().item())


def cosine_logits(base_logits: torch.Tensor, new_logits: torch.Tensor) -> float:
    return float(F.cosine_similarity(
        base_logits.flatten().float().unsqueeze(0),
        new_logits.flatten().float().unsqueeze(0),
        dim=1,
    ).item())


def print_teacher_forced(tokenizer: Any, logits: torch.Tensor, max_tokens: int, title: str) -> None:
    pred = logits.argmax(dim=-1)[0].tolist()[:max_tokens]
    print(f"\n{title}")
    print("-" * 60)
    print(tokenizer.decode(pred, skip_special_tokens=False))


def cosine_gap(embed_tokens: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ea = embed_tokens[a].float()
    eb = embed_tokens[b].float()
    cos = F.cosine_similarity(ea, eb, dim=-1)
    return 1.0 - cos


# =============================================================================
# Main
# =============================================================================

def run() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.agent.adapters import SemanticTokenCodec
    from src.router.kernel import RouterKernel

    torch.set_num_threads(CFG.torch_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(CFG.torch_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(CFG.torch_num_threads)

    device = torch.device(CFG.device)

    print("=" * 70)
    print("GyroSpectacles v10")
    print("=" * 70)
    print(f"device={CFG.device} dtype={CFG.dtype} threads={CFG.torch_num_threads}")
    sys.stdout.flush()

    # Check artifacts
    print("\n[1] Checking artifacts...")
    missing = []
    for name, path in [
        ("act_adapter", CFG.act_adapter_path),
        ("bias_head", CFG.bias_head_path),
        ("probe", CFG.probe_path),
        ("codec", CFG.codec_path),
        ("phenomenology", CFG.atlas_dir / "phenomenology.npz"),
    ]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("ERROR: Missing files:")
        for m in missing:
            print(f"  {m}")
        return
    print("    All artifacts OK")
    sys.stdout.flush()

    # Load model
    print("\n[2] Loading model...")
    sys.stdout.flush()
    gc.collect()
    
    tok = AutoTokenizer.from_pretrained(CFG.model_path)
    print("    Tokenizer OK")
    sys.stdout.flush()
    
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_path,
        torch_dtype=CFG.dtype,
        device_map={"": CFG.device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.output_hidden_states = True
    print("    Model OK")
    sys.stdout.flush()

    embed_tokens = model.model.embed_tokens.weight.detach().float().cpu()
    print(f"    Embeddings: {embed_tokens.shape}")

    # Load components
    print("\n[3] Loading components...")
    lens = load_atlas_lens(CFG.atlas_dir, K=CFG.K)
    print("    Atlas OK")
    
    codec = SemanticTokenCodec.load(CFG.codec_path)
    print("    Codec OK")

    act = ActAdapter(hidden_dim=CFG.hidden_dim, K=CFG.K, hidden_m=256).to(device)
    act.load_state_dict(torch.load(CFG.act_adapter_path, map_location=device, weights_only=True))
    act.eval()
    print("    ActAdapter OK")

    bias_head = BiasHead(hidden_dim=CFG.hidden_dim, hidden_m=256, bias_max=CFG.bias_max).to(device)
    bias_head.load_state_dict(torch.load(CFG.bias_head_path, map_location=device, weights_only=True))
    bias_head.eval()
    print("    BiasHead OK")

    probe = HorizonVertexPhaseProbe(hidden_dim=CFG.hidden_dim, hidden_m=512).to(device)
    probe.load_state_dict(torch.load(CFG.probe_path, map_location=device, weights_only=True))
    probe.eval()
    print("    Probe OK")
    sys.stdout.flush()

    planner = GyroPlanner(CFG, lens, embed_tokens=embed_tokens, codec=codec, device=device)
    kernel = RouterKernel(CFG.atlas_dir)
    print("    Planner/Kernel OK")

    # Run prompts
    all_stats: List[Dict[str, float]] = []

    for pi, prompt in enumerate(TEST_PROMPTS, start=1):
        print("\n" + "=" * 70)
        print(f"PROMPT {pi}/{len(TEST_PROMPTS)}")
        print("=" * 70)

        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=CFG.max_prompt_tokens)
        input_ids = enc["input_ids"].to(device)
        token_ids = input_ids[0].tolist()
        S = len(token_ids)
        print(f"Tokens: {S}")

        print("Forward pass...", end=" ", flush=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=input_ids, output_hidden_states=True)
        t_fwd = (time.perf_counter() - t0) * 1000.0
        print(f"{t_fwd:.0f} ms")

        hs = out.hidden_states
        assert hs is not None, "hidden_states is None"
        hidden_last = hs[-1][0, :S, :].to(torch.float32)
        hidden_gate = hs[CFG.probe_layer_for_gate + 1][0, :S, :].to(torch.float32)

        base_logits = out.logits[0, :S, :].float().cpu()
        base_top1 = base_logits.argmax(dim=-1)
        _, base_top5 = base_logits.topk(5, dim=-1)

        # Compute gyro targets
        kernel.reset()
        planner.reset()

        gyro_targets = torch.empty((S,), dtype=torch.int64)
        h_after = torch.empty((S,), dtype=torch.int64)
        chi_after = torch.empty((S,), dtype=torch.int64)
        p_after = torch.empty((S,), dtype=torch.int64)

        bytes4 = [codec.encode(int(tid)) for tid in token_ids]
        gamma = lens.gamma_table
        bw = lens.byte_weight

        print("Computing gyro targets...", end=" ", flush=True)
        t0 = time.perf_counter()
        
        for t in range(S):
            idx0 = int(kernel.state_index)
            h0 = int(kernel.current_horizon)
            chi0 = int(kernel.current_vertex)
            p0 = int(kernel.current_phase)

            a_curr = act(hidden_last[t].to(device))
            planned = planner.plan_bytes(idx0, h0, chi0, p0, a_curr)
            gyro_targets[t] = planner.decode_token(planned, hidden_last[t])

            chi_prev = int(kernel.current_vertex)
            for b in bytes4[t]:
                chi_next = int(kernel.peek_next_vertex(b))
                w = int(bw[int(b)])
                g = float(gamma[chi_prev, chi_next, w])
                planner.memory.update(int(kernel.current_horizon), int(kernel.current_phase), a_curr, g)
                kernel.step_byte(b)
                chi_prev = chi_next

            h_after[t] = int(kernel.current_horizon)
            chi_after[t] = int(kernel.current_vertex)
            p_after[t] = int(kernel.current_phase)

        t_gyro = (time.perf_counter() - t0) * 1000.0
        print(f"{t_gyro:.0f} ms")

        # Compute gap and gating
        gap = cosine_gap(embed_tokens, base_top1.cpu(), gyro_targets.cpu())

        with torch.inference_mode():
            lh, lv, lp = probe(hidden_gate.to(device))
            ph = F.softmax(lh, dim=-1)
            pv = F.softmax(lv, dim=-1)
            pp = F.softmax(lp, dim=-1)

            idx_t = torch.arange(S, device=device)
            mh = 1.0 - ph[idx_t, h_after.to(device)]
            mv = 1.0 - pv[idx_t, chi_after.to(device)]
            mp = 1.0 - pp[idx_t, p_after.to(device)]
            mismatch = ((mh + mv + mp) / 3.0).float().cpu()

        eff = gap + CFG.probe_mismatch_weight * mismatch
        gate = torch.zeros_like(eff)
        gate[eff > CFG.gap_high] = 1.0
        gate[(eff > CFG.gap_low) & (eff <= CFG.gap_high)] = 0.5

        with torch.inference_mode():
            bias = bias_head(hidden_last.to(device)).float().cpu()

        boosted_logits = base_logits.clone()
        boosted_logits[torch.arange(S), gyro_targets] += gate * bias

        boost_top1 = boosted_logits.argmax(dim=-1)
        _, boost_top5 = boosted_logits.topk(5, dim=-1)

        # Metrics
        base_hit1 = float((base_top1 == gyro_targets).float().mean().item())
        base_hit5 = float((base_top5 == gyro_targets.unsqueeze(-1)).any(dim=-1).float().mean().item())
        boost_hit1 = float((boost_top1 == gyro_targets).float().mean().item())
        boost_hit5 = float((boost_top5 == gyro_targets.unsqueeze(-1)).any(dim=-1).float().mean().item())

        intervention_rate = float((gate > 0).float().mean().item())
        avg_gap = float(gap.mean().item())
        avg_mismatch = float(mismatch.mean().item())

        base_logits_b = base_logits.unsqueeze(0)
        boosted_logits_b = boosted_logits.unsqueeze(0)

        kld = kl_div_fn(base_logits_b, boosted_logits_b)
        cosL = cosine_logits(base_logits_b, boosted_logits_b)
        nll_b = nll_next_token(base_logits_b, input_ids.cpu())
        nll_g = nll_next_token(boosted_logits_b, input_ids.cpu())
        nll_ratio = nll_g / max(nll_b, 1e-9)

        print("\nMetrics")
        print("-" * 60)
        print(f"Gyro hit@1: base {base_hit1:.3f} -> boosted {boost_hit1:.3f}")
        print(f"Gyro hit@5: base {base_hit5:.3f} -> boosted {boost_hit5:.3f}")
        print(f"gap={avg_gap:.3f} mismatch={avg_mismatch:.3f} intervention={intervention_rate:.3f}")
        print(f"KL={kld:.4f} cosine={cosL:+.4f} NLL_ratio={nll_ratio:.4f}")

        print_teacher_forced(tok, base_logits_b, CFG.teacher_forced_max_tokens, "BASE argmax")
        print_teacher_forced(tok, boosted_logits_b, CFG.teacher_forced_max_tokens, "BOOST argmax")

        all_stats.append({
            "base_hit1": base_hit1, "base_hit5": base_hit5,
            "boost_hit1": boost_hit1, "boost_hit5": boost_hit5,
            "avg_gap": avg_gap, "avg_mismatch": avg_mismatch,
            "intervention_rate": intervention_rate,
            "kl": kld, "cosine": cosL, "nll_ratio": nll_ratio,
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k in all_stats[0].keys():
        vals = [s[k] for s in all_stats]
        print(f"{k:18s}: {np.mean(vals):.4f}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()