# scripts/gyro_experiments.py
"""
Gyro Experiments — one-run Router-native porting laboratory (CPU-first)

Scope:
- External-only interventions; do NOT edit modeling_bolmo.py.
- Keep GyroLabe; use it as safe override mechanism.
- Do NOT use K43 (features_K43).
- Ground all logic in RouterKernel observables & GF(2) mask-code structure.

Outputs:
- Print-only. No file I/O. No flags.
- One run executes all blocks and prints a final comparative table.
- Keep separator lines (repeated "=") <= 5 total.

Suite (consolidated toward ASI architecture):
B0: uncoupled (no GyroLabe)
B1: local-decoder masking (GyroLabe)

V_ASI_Core+M:
  - Selection rules (forbid kernel-forbidden phase transitions)
  - Lagrangian (depth-4 xyxy closure reward + gentle P_code friction)
  - Holographic boundary law (strict: fused only if horizon or closure)
  - plus local mask

V_Proj_Bulk+M:
  - projector onto 24-bit bulk (u,v) (no ASI controller), plus local mask

V_Proj_Gauge+M:
  - projector, but "gauge preservation" is implemented as *no correction* on specified bits (delta_u bit = 0)
  - erase bits: dup bits in u and v => [8..11, 20..23]
  - plus local mask

V_ASI_Full+M:
  - ASI core controller + bulk projector + local mask

Long runs (2048):
- B0_LONG, B1_LONG
- V_ASI_Core+M_LONG
- V_ASI_Full+M_LONG
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings(
    "ignore", message=".*rope_config_validation.*",
    category=FutureWarning, module="transformers.*",
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_B12,
    LAYER_MASK_12,
    mask12_for_byte,
    popcount,
    unpack_state,
    vertex_charge_from_mask,
)
from src.router.kernel import RouterKernel
from src.tools.gyrolabe import CouplingConfig, GyroLabe, detect_device


# ---------------------------------------------------------------------------
# Bolmo loading and patches (external-only)
# ---------------------------------------------------------------------------

def _patch_rope_default_for_bolmo() -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        if hasattr(config, "standardize_rope_params"):
            config.standardize_rope_params()
        rp = getattr(config, "rope_parameters", None)
        if rp is not None:
            rp = rp.get(layer_type, rp) if layer_type and isinstance(rp, dict) else rp
            if isinstance(rp, dict):
                base = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
                partial = rp.get("partial_rotary_factor", 1.0)
            else:
                base = getattr(config, "rope_theta", 10000.0)
                partial = 1.0
        else:
            base = getattr(config, "rope_theta", 10000.0)
            partial = 1.0
        base = float(base.item()) if hasattr(base, "item") else float(base)
        partial = float(partial.item()) if hasattr(partial, "item") else float(partial)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim))
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    from transformers.modeling_utils import PreTrainedModel
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE

    if getattr(PreTrainedModel._init_weights, "__bolmo_rope_patch__", False):
        return

    _orig = PreTrainedModel._init_weights

    def _patched(self, module):
        if (
            "RotaryEmbedding" in module.__class__.__name__
            and getattr(module, "original_inv_freq", None) is not None
            and getattr(module, "rope_type", None) == "default"
            and "default" in _ROPE
        ):
            buf, _ = _ROPE["default"](module.config, None, None, None)
            with torch.no_grad():
                b = buf.to(module.inv_freq.device, dtype=module.inv_freq.dtype)
                module.inv_freq.copy_(b)
                module.original_inv_freq.copy_(b)
            return
        _orig(self, module)

    _patched.__bolmo_rope_patch__ = True
    PreTrainedModel._init_weights = _patched


def _patch_generation_prepare() -> None:
    from transformers.generation.utils import GenerationMixin

    if getattr(GenerationMixin._prepare_generation_config, "__bolmo_patch__", False):
        return
    _orig = GenerationMixin._prepare_generation_config

    def _patched(self, generation_config=None, *args, **kwargs):
        try:
            return _orig(self, generation_config, *args, **kwargs)
        except TypeError as e:
            if "takes 2 positional arguments" in str(e) and len(args) > 0:
                return _orig(self, generation_config, **kwargs)
            raise

    _patched.__bolmo_patch__ = True
    GenerationMixin._prepare_generation_config = _patched


def ensure_dolma2_tokenizer(local_dir: Path) -> Path:
    needed = ["tokenizer.json", "tokenizer_config.json"]
    if all((local_dir / name).exists() for name in needed):
        return local_dir
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(f"dolma2 tokenizer not cached at {local_dir} and offline mode is enabled")
    local_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="allenai/dolma2-tokenizer",
        local_dir=str(local_dir),
        allow_patterns=["tokenizer.json", "tokenizer_config.json"],
    )
    return local_dir


def bolmo_reset_local_caches(model: Any) -> None:
    try:
        model.model.local_encoder.free_inference_cache()
    except Exception:
        pass
    try:
        model.model.local_decoder.free_inference_cache()
    except Exception:
        pass


def load_bolmo(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _patch_rope_default_for_bolmo()
    _patch_generation_prepare()

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading {model_dir.name} on {device} (dtype={torch_dtype})...")

    model_dir = model_dir.resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    from configuration_bolmo import BolmoConfig  # type: ignore
    from modeling_bolmo import BolmoForCausalLM  # type: ignore

    config = BolmoConfig.from_pretrained(model_dir, local_files_only=True)
    model = BolmoForCausalLM.from_pretrained(
        model_dir, config=config, local_files_only=True, torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    dolma2_dir = ensure_dolma2_tokenizer(Path("data/models/dolma2-tokenizer"))
    tc = model.model.tokenizer_config
    tc.original_identifier = str(dolma2_dir)
    tokenizer = tc.build()

    tokenizer.config.eos_token_id = 2
    try:
        model.config.eos_token_id = 2
    except Exception:
        pass

    print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# MaskState (MUST match what modeling_bolmo expects)
# ---------------------------------------------------------------------------

class MaskState:
    """
    modeling_bolmo.py expects boundary_state and pad_state objects
    with selective_get/selective_put/selective_add semantics.
    """
    def __init__(self, mask: torch.Tensor):
        self.mask = mask
        self.inv_mask = ~mask
        self.cpu_mask = mask.detach().cpu()
        self._all = bool(self.cpu_mask.all().item())
        self._any = bool(self.cpu_mask.any().item())

    def any(self) -> bool:
        return self._any

    def all(self) -> bool:
        return self._all

    def selective_get(self, x: torch.Tensor, inv: bool = False) -> torch.Tensor:
        if inv:
            if self.all():
                return x[[]]
            if not self.any():
                return x
            return x[self.inv_mask]
        else:
            if self.all():
                return x
            if not self.any():
                return x[[]]
            return x[self.mask]

    def selective_put(self, x: torch.Tensor, out: torch.Tensor, inv: bool = False) -> None:
        if inv:
            if self.all():
                return
            if not self.any():
                out.copy_(x)
                return
            out[self.inv_mask] = x
        else:
            if self.all():
                out.copy_(x)
                return
            if not self.any():
                return
            out[self.mask] = x

    def selective_add(self, x: int, out: torch.Tensor, inv: bool = False) -> None:
        if inv:
            if self.all():
                return
            if not self.any():
                out.add_(x)
                return
            out[self.inv_mask] += x
        else:
            if self.all():
                out.add_(x)
                return
            if not self.any():
                return
            out[self.mask] += x


# ========
# Physics tables (no K43)
# ========

def byte_to_intron(byte: int) -> int:
    return (byte & 0xFF) ^ 0xAA


def intron_micro(intron: int) -> int:
    return intron & 0x3F


def token_to_byte_and_fused(token_id: int, token_offset: int) -> tuple[int | None, bool]:
    """
    Bolmo tokenization:
      token_offset = 4 (specials first)
      [4..259] plain byte tokens  -> byte = tok - 4
      [260..515] fused boundary   -> byte = tok - 260
    """
    t = int(token_id)
    if t < token_offset:
        return (None, False)
    if token_offset <= t < token_offset + 256:
        return (t - token_offset, False)
    if token_offset + 256 <= t < token_offset + 512:
        return (t - (token_offset + 256), True)
    return (None, False)


def build_physics_tables() -> dict[str, np.ndarray]:
    mask12 = np.zeros(256, dtype=np.uint16)
    weight = np.zeros(256, dtype=np.uint8)
    chi = np.zeros(256, dtype=np.uint8)
    intr = np.zeros(256, dtype=np.uint8)
    micro = np.zeros(256, dtype=np.uint8)

    for b in range(256):
        m = mask12_for_byte(b)
        mask12[b] = np.uint16(m)
        weight[b] = np.uint8(popcount(m))
        chi[b] = np.uint8(vertex_charge_from_mask(m))

        x = byte_to_intron(b)
        intr[b] = np.uint8(x)
        micro[b] = np.uint8(intron_micro(x))

    dist = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        mi = int(mask12[i])
        for j in range(256):
            dist[i, j] = np.uint8(popcount(mi ^ int(mask12[j])))

    counts = np.bincount(dist.reshape(-1), minlength=13).astype(np.float64)
    P_code = counts / counts.sum()

    return {
        "mask12_by_byte": mask12,
        "weight_by_byte": weight,
        "chi_by_byte": chi,
        "intron_by_byte": intr,
        "micro_by_byte": micro,
        "code_dist_256": dist,
        "P_code": P_code,
    }


# ========
# Ridge regression (no sklearn)
# ========

@dataclass
class RidgeModel:
    W: np.ndarray
    b: np.ndarray


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> RidgeModel:
    n, d = X.shape
    X1 = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    I = np.eye(d + 1, dtype=X.dtype)
    I[-1, -1] = 0.0
    A = X1.T @ X1 + lam * I
    B = X1.T @ Y
    W1 = np.linalg.solve(A, B)
    return RidgeModel(W=W1[:d, :], b=W1[d, :])


def ridge_predict(m: RidgeModel, X: np.ndarray) -> np.ndarray:
    return X @ m.W + m.b


def r2_score(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    eps = 1e-12
    sse = np.sum((Y_true - Y_pred) ** 2, axis=0)
    sst = np.sum((Y_true - Y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + eps
    return float(np.mean(1.0 - sse / sst))


# ========
# Kernel feature extraction
# ========

def kernel_bulk_coordinates_sign(kernel: RouterKernel) -> np.ndarray:
    idx = int(kernel.state_index[0])
    s24 = int(kernel.ontology[idx])
    a12, b12 = unpack_state(s24)
    u = (a12 ^ ARCHETYPE_A12) & LAYER_MASK_12
    v = (b12 ^ ARCHETYPE_B12) & LAYER_MASK_12

    bits = np.empty(24, dtype=np.float64)
    for i in range(12):
        bits[i] = 1.0 if ((u >> i) & 1) else -1.0
        bits[12 + i] = 1.0 if ((v >> i) & 1) else -1.0
    return bits


def kernel_horizon_distance(kernel: RouterKernel) -> int:
    idx = int(kernel.state_index[0])
    s24 = int(kernel.ontology[idx])
    a12, b12 = unpack_state(s24)
    return popcount(a12 ^ (b12 ^ 0xFFF))


def _kernel_current_phase(kernel: RouterKernel) -> int:
    # Prefer explicit current_phase if present
    if hasattr(kernel, "current_phase"):
        try:
            return int(kernel.current_phase[0])
        except Exception:
            pass
    # Fallback to table-based definition used in earlier scripts
    idx = int(kernel.state_index[0])
    lb = int(kernel.last_byte[0]) & 0xFF
    if hasattr(kernel, "phase"):
        try:
            return int(kernel.phase[idx, lb])
        except Exception:
            pass
    if hasattr(kernel, "state_phase"):
        try:
            return int(kernel.state_phase[idx])
        except Exception:
            pass
    return 0


def _kernel_next_phase_by_byte(kernel: RouterKernel, idx: int) -> np.ndarray:
    arr = getattr(kernel, "next_phase", None)
    if arr is None:
        raise AttributeError("RouterKernel has no next_phase table; cannot enforce selection rules.")
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        return arr[idx, :]
    if isinstance(arr, np.ndarray) and arr.ndim == 1:
        return arr
    # last resort: try indexing like a tensor
    try:
        return np.asarray(arr[idx, :])
    except Exception as e:
        raise TypeError(f"Unsupported next_phase table type/shape: {type(arr)}") from e


# ========
# Projector hook
# ========

@dataclass
class Projector:
    layer_idx: int
    fwd: RidgeModel
    back: RidgeModel
    alpha: float
    erase_bits: Optional[list[int]] = None
    use_complement: bool = False


class ProjectorHook:
    def __init__(self, kernel: RouterKernel, proj: Projector):
        self.kernel = kernel
        self.proj = proj

    def __call__(self, module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            return output
        B, S, _D = output.shape
        if B != 1 or S != 1:
            return output

        h = output[0, 0, :].detach().cpu().numpy().astype(np.float64)[None, :]

        u_true = kernel_bulk_coordinates_sign(self.kernel)[None, :]
        if self.proj.use_complement:
            u_true = -u_true

        u_pred = ridge_predict(self.proj.fwd, h)
        delta_u = u_true - u_pred

        # Gauge preservation: do NOT correct these bits (leave model's own value)
        if self.proj.erase_bits:
            for bit in self.proj.erase_bits:
                if 0 <= bit < 24:
                    delta_u[0, bit] = 0.0

        delta_h = ridge_predict(self.proj.back, delta_u)
        h_new = h + float(self.proj.alpha) * delta_h

        out = output.clone()
        out[0, 0, :] = torch.from_numpy(h_new[0]).to(out.device, out.dtype)
        return out


# ========
# ASI Core Controller (logit modifier)
# ========

@dataclass
class LogitMods:
    temperature: float = 0.7
    top_k: int = 40

    asi_core: bool = False

    # parameters for ASI controller (kept explicit for transparency)
    orbit_reward: float = 1.5
    orbit_break_penalty: float = 0.5
    friction_scale: float = 0.05

    # boundary law: strict fused gating
    fuse_reward: float = 1.0
    fuse_forbid: bool = True  # if True, fused tokens are -inf unless allowed

    depth4_history: list[int] = field(default_factory=list)


def apply_logit_mods(
    kernel: RouterKernel,
    mods: LogitMods,
    scores: torch.Tensor,
    tables: dict[str, np.ndarray],
    token_byte: np.ndarray,
    token_fused: np.ndarray,
) -> torch.Tensor:
    if not mods.asi_core:
        return scores

    out = scores.clone()

    idx = int(kernel.state_index[0])
    s24 = int(kernel.ontology[idx])
    a12, b12 = unpack_state(s24)
    
    horizon_dist = popcount(a12 ^ (b12 ^ 0xFFF))
    on_horizon = (horizon_dist == 0)
    
    chi_cur = int(kernel.state_vertex[idx])
    p_cur = _kernel_current_phase(kernel)
    
    next_chi_by_byte = kernel.next_vertex[idx, :]
    next_p_by_byte = _kernel_next_phase_by_byte(kernel, idx)
    
    dist256 = tables["code_dist_256"]
    last_b = int(kernel.last_byte[0]) & 0xFF
    d_action_by_byte = dist256[last_b, np.arange(256, dtype=np.int64)]

    history = mods.depth4_history[-3:] if len(mods.depth4_history) >= 3 else []
    
    for t in range(out.numel()):
        b = int(token_byte[t])
        if b < 0:
            continue
            
        chi_next = int(next_chi_by_byte[b])
        p_next = int(next_p_by_byte[b])
        is_fused = (int(token_fused[t]) == 1)

        # 1. Soft Selection Rules (Replacing -inf ban)
        # Penalize, rather than forbid, kernel-unlikely phase transitions
        if (p_cur == 0 and p_next == 0) or (p_cur == 3 and p_next == 1):
            out[t] -= 0.5

        # 2. Lagrangian Micro-Cost
        closes_orbit = False
        if len(history) == 3:
            x, y, x2 = history
            if x == x2 and y != x:
                if b == y:
                    closes_orbit = True
                    out[t] += float(mods.orbit_reward)
                else:
                    out[t] -= float(mods.orbit_break_penalty)
                    
        if not closes_orbit:
            d = int(d_action_by_byte[b])
            out[t] -= float(mods.friction_scale) * math.log(float(tables["P_code"][d]) + 1e-12)

        # 3. Soft Holographic Boundary Law
        # Restore baseline token generation mechanics, with geometric nudges
        if is_fused:
            # Baseline small penalty to prevent runaway patch chopping
            out[t] -= 0.1
            
            # Boost if physics align
            if on_horizon or closes_orbit:
                out[t] += float(mods.fuse_reward)

    return out


# ========
# Decoding loop
# ========

@dataclass
class RunResult:
    name: str
    elapsed: float
    tok_s: float
    n_steps: int
    n_bytes: int
    mean_raw_logprob: float
    mean_adj_logprob: float
    unique_bytes: int
    fused_rate: float
    mean_patch_len: float
    median_patch_len: float
    max_patch_len: int
    mean_code_dist: float
    kl_cd_millibits: float
    unique_h: int
    h_entropy: float
    chi_hist: list[int]
    phase_hist: list[int]
    sample_text: str = ""
    horizon_trajectory: str = ""


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = (p > 0) & (q > 0)
    return float(np.sum(p[m] * np.log2((p[m] + eps) / (q[m] + eps))))


@torch.inference_mode()
def bolmo_generate_one(
    model,
    tokenizer,
    atlas_dir: Path,
    prompt: str,
    max_new_tokens: int,
    labe: GyroLabe | None,
    mods: LogitMods,
    tables: dict[str, np.ndarray],
    token_byte: np.ndarray,
    token_fused: np.ndarray,
    projector: Projector | None = None,
) -> RunResult:
    # reproducibility per-variant
    torch.manual_seed(42)
    np.random.seed(42)

    bolmo_reset_local_caches(model)
    mods.depth4_history.clear()

    device = next(model.parameters()).device
    token_offset = int(getattr(tokenizer, "offset", 4))
    boundary_offset = token_offset + 256
    bpe_end_id = int(tokenizer.bpe_token_end_id)

    if labe is not None:
        labe.reset()
        kernel = labe.kernel
    else:
        kernel = RouterKernel(atlas_dir=atlas_dir)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids_full = enc["input_ids"].to(device)
    prompt_ids = input_ids_full[0].tolist()

    def prime_kernel_with_tokens(toks: list[int]) -> None:
        for t in toks:
            b, _f = token_to_byte_and_fused(t, token_offset)
            if b is None:
                continue
            if labe is not None:
                labe.advance_with_token(b)
            else:
                kernel.step_byte(int(b) & 0xFF)

    expand_input_ids = getattr(model.model.local_encoder, "add_expanded_embeddings", False)
    if expand_input_ids:
        expanded_prompt = torch.tensor(
            model.model.tokenizer.expand_byte_ids(prompt_ids),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
    else:
        expanded_prompt = None

    pad_id = int(tokenizer.pad_token_id)
    sequence_start_indices = (input_ids_full == pad_id).sum(-1)

    boundary_mask = model.model.prefill_boundary_prediction_forward(
        input_ids_full,
        expanded_input_ids=expanded_prompt,
        sequence_start_indices=sequence_start_indices,
    )

    model.model.local_encoder.free_inference_cache()
    model.model.local_decoder.free_inference_cache()
    model.model.local_encoder.prepare_inference_cache(batch_size=1)
    model.model.local_decoder.prepare_inference_cache(batch_size=1)

    # roll back by one (lookahead=1)
    boundary_mask = boundary_mask[:, :-1]
    forced_decoding_id = prompt_ids[-1] if len(prompt_ids) > 0 else None
    input_ids = input_ids_full[:, :-1]
    if expanded_prompt is not None:
        expanded_prompt = expanded_prompt[:, :-1]
    sequence_start_indices = (input_ids == pad_id).sum(-1)

    prime_kernel_with_tokens(input_ids[0].tolist())

    boundary_state = MaskState(boundary_mask[:, -1].clone())
    pad_state = MaskState(torch.zeros((1,), dtype=torch.bool, device=device))

    finished = torch.zeros((1,), dtype=torch.bool, device=device)
    eos = int(getattr(tokenizer, "eos_token_id", -1))

    generated = input_ids.clone()
    non_boundary_generated_tokens = [int(input_ids[0, -1].item())] if input_ids.numel() > 0 else []
    global_past_key_values = None
    is_first_forward = True

    hook_handle = None
    if projector is not None:
        layer_mod = model.model.local_decoder.layers[projector.layer_idx]
        hook_handle = layer_mod.register_forward_hook(ProjectorHook(kernel, projector))

    raw_logprobs: list[float] = []
    adj_logprobs: list[float] = []
    emitted_bytes: list[int] = []
    emitted_fused: list[bool] = []

    t0 = time.perf_counter()

    for _step in range(max_new_tokens):
        if labe is not None:
            labe.begin_step()

        if is_first_forward:
            input_ids_for_model = generated
            expanded_for_model = expanded_prompt
            boundary_mask_for_model = boundary_mask
        else:
            last_tok = non_boundary_generated_tokens[-1]
            input_ids_for_model = torch.tensor([[last_tok]], device=device, dtype=generated.dtype)

            if expand_input_ids:
                expanded_last = model.model.tokenizer.expand_byte_ids(generated[0].tolist(), n_last=1)
                expanded_for_model = torch.tensor([expanded_last], device=device, dtype=torch.long)
            else:
                expanded_for_model = None

            boundary_mask_for_model = None

        out = model(
            input_ids_for_model,
            expanded_input_ids=expanded_for_model,
            boundary_mask=boundary_mask_for_model,
            boundary_state=boundary_state,
            pad_state=pad_state,
            sequence_start_indices=sequence_start_indices,
            logits_to_keep=1,
            use_cache=True,
            past_key_values=global_past_key_values,
        )
        global_past_key_values = out.past_key_values

        if labe is not None:
            labe.end_step()

        raw_logits = out.logits[0, -1].float()
        raw_lp = torch.log_softmax(raw_logits, dim=-1)

        scores = raw_logits / max(float(mods.temperature), 1e-8)
        scores = apply_logit_mods(
            kernel=kernel,
            mods=mods,
            scores=scores,
            tables=tables,
            token_byte=token_byte,
            token_fused=token_fused,
        )

        # if everything got killed (should be rare), revert to raw
        if torch.isneginf(scores).all() or torch.isnan(scores).any():
            scores = raw_logits / max(float(mods.temperature), 1e-8)

        # forced decoding for first step (keep fused vs non-fused option)
        if forced_decoding_id is not None and is_first_forward:
            forced = int(forced_decoding_id)
            forced_fused = forced + boundary_offset
            keep = torch.full_like(scores, float("-inf"))
            if 0 <= forced < keep.numel():
                keep[forced] = scores[forced]
            if 0 <= forced_fused < keep.numel():
                keep[forced_fused] = scores[forced_fused]
            scores = keep
            forced_decoding_id = None
            if torch.isneginf(scores).all() or torch.isnan(scores).any():
                scores = raw_logits / max(float(mods.temperature), 1e-8)

        # top-k sampling
        if mods.top_k > 0:
            k = min(int(mods.top_k), scores.numel())
            topv, topi = torch.topk(scores, k=k)
            probs = torch.softmax(topv, dim=-1)
            sample_idx = torch.multinomial(probs, 1).item()
            next_tok = int(topi[sample_idx].item())
            adj_lp = torch.log_softmax(topv, dim=-1)[sample_idx].item()
        else:
            probs = torch.softmax(scores, dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())
            adj_lp = torch.log_softmax(scores, dim=-1)[next_tok].item()

        raw_logprobs.append(float(raw_lp[next_tok].item()))
        adj_logprobs.append(float(adj_lp))

        b, fused = token_to_byte_and_fused(next_tok, token_offset)
        if b is not None and next_tok != bpe_end_id:
            emitted_bytes.append(int(b))
            emitted_fused.append(bool(fused))
            non_boundary_generated_tokens.append(int(b))
            if labe is not None:
                labe.advance_with_token(int(b))
            else:
                kernel.step_byte(int(b))
            mods.depth4_history.append(int(b))
            if len(mods.depth4_history) > 4:
                mods.depth4_history.pop(0)

        next_tokens = torch.tensor([next_tok], device=device, dtype=torch.long)
        boundary_state = MaskState((next_tokens == bpe_end_id) | (next_tokens >= boundary_offset) | finished)
        pad_state = MaskState((next_tokens == bpe_end_id) | finished)

        generated = torch.cat([generated, next_tokens.view(1, 1).to(generated.dtype)], dim=1)
        is_first_forward = False

        if eos >= 0 and next_tok == eos:
            break

    elapsed = time.perf_counter() - t0

    if hook_handle is not None:
        hook_handle.remove()

    # patch lengths (fused marks patch end)
    patch_lens: list[int] = []
    if emitted_bytes:
        cur = 0
        for is_fused in emitted_fused:
            cur += 1
            if is_fused:
                patch_lens.append(cur)
                cur = 0
        if cur > 0:
            patch_lens.append(cur)

    mean_patch_len = float(np.mean(patch_lens)) if patch_lens else 0.0
    median_patch_len = float(np.median(patch_lens)) if patch_lens else 0.0
    max_patch_len = int(np.max(patch_lens)) if patch_lens else 0

    # code-distance KL vs P_code
    if len(emitted_bytes) >= 2:
        m = tables["mask12_by_byte"]
        dseq = [popcount(int(m[emitted_bytes[i]]) ^ int(m[emitted_bytes[i + 1]])) for i in range(len(emitted_bytes) - 1)]
        d_counts = np.bincount(np.array(dseq, dtype=np.int64), minlength=13).astype(np.float64)
        kl = kl_bits(d_counts, tables["P_code"]) * 1000.0
        mean_cd = float(np.mean(dseq))
    else:
        kl = 0.0
        mean_cd = 0.0

    # replay kernel for interpretables
    replay = RouterKernel(atlas_dir=atlas_dir)
    for t in input_ids[0].tolist():
        bb, _ = token_to_byte_and_fused(int(t), token_offset)
        if bb is not None:
            replay.step_byte(int(bb))

    h_list: list[int] = []
    chi_list: list[int] = []
    p_list: list[int] = []
    for bb in emitted_bytes:
        replay.step_byte(int(bb))
        h_list.append(int(replay.current_horizon[0]))
        chi_list.append(int(replay.current_vertex[0]))
        p_list.append(int(replay.current_phase[0]))

    h_counts = np.bincount(np.array(h_list, dtype=np.int64), minlength=256)
    chi_counts = np.bincount(np.array(chi_list, dtype=np.int64), minlength=4)
    p_counts = np.bincount(np.array(p_list, dtype=np.int64), minlength=4)

    unique_h = int(np.count_nonzero(h_counts))
    h_ent = entropy_from_counts(h_counts)

    n_bytes = len(emitted_bytes)
    unique_bytes_count = int(len(set(emitted_bytes)))
    fused_rate = float(np.mean(emitted_fused)) if emitted_fused else 0.0
    mean_raw_lp = float(np.mean(raw_logprobs)) if raw_logprobs else 0.0
    mean_adj_lp = float(np.mean(adj_logprobs)) if adj_logprobs else 0.0

    # capture more text (qualitative inspection)
    if generated.numel() > 0:
        max_text = 1200 if max_new_tokens <= 512 else 500
        sample_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)[:max_text]
    else:
        sample_text = "[empty]"

    horizon_traj = " ".join([f"{h:02x}" for h in h_list[:80]])

    bolmo_reset_local_caches(model)
    return RunResult(
        name="",
        elapsed=elapsed,
        tok_s=(n_bytes / max(elapsed, 1e-9)),
        n_steps=max_new_tokens,
        n_bytes=n_bytes,
        mean_raw_logprob=mean_raw_lp,
        mean_adj_logprob=mean_adj_lp,
        unique_bytes=unique_bytes_count,
        fused_rate=fused_rate,
        mean_patch_len=mean_patch_len,
        median_patch_len=median_patch_len,
        max_patch_len=max_patch_len,
        mean_code_dist=mean_cd,
        kl_cd_millibits=float(kl),
        unique_h=unique_h,
        h_entropy=h_ent,
        chi_hist=chi_counts.tolist(),
        phase_hist=p_counts.tolist(),
        sample_text=sample_text,
        horizon_trajectory=horizon_traj,
    )


# ========
# Calibration
# ========

@dataclass
class Calibration:
    emb_to_mask: RidgeModel
    emb_to_chi: RidgeModel
    runtime_best_layer: int
    hid_to_bulk: RidgeModel
    bulk_to_hid: RidgeModel
    use_complement: bool
    r2_direct: float
    r2_complement: float


@torch.inference_mode()
def run_calibration(
    model,
    tokenizer,
    atlas_dir: Path,
    tables: dict[str, np.ndarray],
    prompt: str,
    steps: int = 256,
) -> Calibration:
    bolmo_reset_local_caches(model)
    device = next(model.parameters()).device
    token_offset = int(getattr(tokenizer, "offset", 4))

    emb_w = model.model.local_encoder.byte_embedding.weight.detach().cpu().numpy().astype(np.float64)
    E = emb_w[token_offset:token_offset + 256, :]

    # emb -> mask12 sign
    Y_mask = np.empty((256, 12), dtype=np.float64)
    for b in range(256):
        m = int(tables["mask12_by_byte"][b])
        for i in range(12):
            Y_mask[b, i] = 1.0 if ((m >> i) & 1) else -1.0
    emb_to_mask = ridge_fit(E, Y_mask, lam=1e-2)
    r2_mask = r2_score(Y_mask, ridge_predict(emb_to_mask, E))

    # emb -> chi onehot
    Y_chi = np.zeros((256, 4), dtype=np.float64)
    for b in range(256):
        c = int(tables["chi_by_byte"][b])
        Y_chi[b, c] = 1.0
    emb_to_chi = ridge_fit(E, Y_chi, lam=1e-2)
    r2_chi = r2_score(Y_chi, ridge_predict(emb_to_chi, E))

    print("Calibration: embedding bridges")
    print(f"  R²(emb -> mask12_sign): {r2_mask:.6f}")
    print(f"  R²(emb -> chi_onehot):  {r2_chi:.6f}")

    print("Calibration: runtime bridge (hidden -> bulk (u,v))")
    steps2 = min(128, steps)

    captured: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    handles = []
    for li in range(4):
        mod = model.model.local_decoder.layers[li]

        def make_hook(k: int):
            def _hook(module, inp, out):
                if isinstance(out, torch.Tensor) and out.ndim == 3 and out.shape[0] == 1:
                    v = out[0, -1, :].detach().cpu().numpy().astype(np.float64)
                    captured[k].append(v)
                return None
            return _hook

        handles.append(mod.register_forward_hook(make_hook(li)))

    enc = tokenizer(prompt, return_tensors="pt")
    ctx = enc["input_ids"].to(device)
    gen_buf = ctx.clone()

    emitted_bytes: list[int] = []
    for _i in range(steps2):
        out = model(gen_buf, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits / 0.7, dim=-1)
        tok = int(torch.multinomial(probs, 1).item())
        b, _fused = token_to_byte_and_fused(tok, token_offset)
        if b is not None:
            emitted_bytes.append(int(b))
        gen_buf = torch.cat([gen_buf, torch.tensor([[tok]], device=device, dtype=gen_buf.dtype)], dim=1)

    for h in handles:
        h.remove()

    n = len(emitted_bytes)
    Y = np.empty((n, 24), dtype=np.float64)

    replay_kernel = RouterKernel(atlas_dir=atlas_dir)
    for t in ctx[0].tolist():
        bb, _ = token_to_byte_and_fused(int(t), token_offset)
        if bb is not None:
            replay_kernel.step_byte(int(bb))

    for t, b in enumerate(emitted_bytes):
        replay_kernel.step_byte(b)
        Y[t, :] = kernel_bulk_coordinates_sign(replay_kernel)

    Y_comp = -Y

    best_layer = 0
    best_r2 = -1.0
    best_fwd = None
    best_back = None
    best_r2_direct = -1.0
    best_r2_comp = -1.0
    use_complement = False

    for li in range(4):
        if len(captured[li]) < n:
            continue
        X = np.stack(captured[li][:n], axis=0)

        fwd_d = ridge_fit(X, Y, lam=1e-2)
        r2_d = r2_score(Y, ridge_predict(fwd_d, X))

        fwd_c = ridge_fit(X, Y_comp, lam=1e-2)
        r2_c = r2_score(Y_comp, ridge_predict(fwd_c, X))

        print(f"  layer {1000 + li}: R²(direct)={r2_d:.4f}  R²(complement)={r2_c:.4f}")

        if r2_c > r2_d:
            eff_r2, eff_fwd, eff_Y, layer_comp = r2_c, fwd_c, Y_comp, True
        else:
            eff_r2, eff_fwd, eff_Y, layer_comp = r2_d, fwd_d, Y, False

        if eff_r2 > best_r2:
            best_r2 = eff_r2
            best_layer = li
            best_fwd = eff_fwd
            best_back = ridge_fit(eff_Y, X, lam=1e-2)
            best_r2_direct = r2_d
            best_r2_comp = r2_c
            use_complement = layer_comp

    assert best_fwd is not None and best_back is not None
    print(f"  Selected layer {1000 + best_layer}: {'complement' if use_complement else 'direct'} mapping")

    return Calibration(
        emb_to_mask=emb_to_mask,
        emb_to_chi=emb_to_chi,
        runtime_best_layer=best_layer,
        hid_to_bulk=best_fwd,
        bulk_to_hid=best_back,
        use_complement=use_complement,
        r2_direct=best_r2_direct,
        r2_complement=best_r2_comp,
    )


# ========
# Printing helpers (clean)
# ========

def print_header() -> None:
    # Separator line #1 (keep total <= 5)
    print("Gyro Experiments — Router-native Bolmo porting lab")
    print("=" * 72)


def format_stats_line(r: RunResult) -> str:
    return (
        f"{r.name:18s}  "
        f"{r.n_bytes:5d}  "
        f"{r.tok_s:6.2f}  "
        f"{r.mean_raw_logprob:8.4f}  "
        f"{r.kl_cd_millibits:9.2f}  "
        f"{r.unique_h:3d}/256  "
        f"H={r.h_entropy:5.2f}  "
        f"cd={r.mean_code_dist:4.2f}  "
        f"fused={100 * r.fused_rate:5.1f}%  "
        f"patch={r.mean_patch_len:4.1f}/{r.median_patch_len:4.1f}/{r.max_patch_len:2d}  "
        f"chi={r.chi_hist}"
    )


def print_run_result(r: RunResult, text_chars: int, horizon_tokens: int = 60) -> None:
    print(format_stats_line(r))

    txt = (r.sample_text or "").replace("\n", " ")
    txt = txt[:text_chars]
    if txt:
        chunk = 240
        for i in range(0, len(txt), chunk):
            print(f"  text[{i:03d}:{min(i + chunk, len(txt)):03d}]: {txt[i:i + chunk]}")

    hz = (r.horizon_trajectory or "").split()
    if hz:
        hz = hz[:horizon_tokens]
        print(f"  horizon[0..{len(hz)-1}]: {' '.join(hz)}")


# ========
# Orchestration
# ========

def build_token_maps(vocab_size: int, token_offset: int) -> tuple[np.ndarray, np.ndarray]:
    token_byte = np.full(vocab_size, -1, dtype=np.int16)
    token_fused = np.zeros(vocab_size, dtype=np.uint8)
    for t in range(vocab_size):
        b, fused = token_to_byte_and_fused(t, token_offset)
        if b is None:
            continue
        token_byte[t] = np.int16(b)
        token_fused[t] = np.uint8(1 if fused else 0)
    return token_byte, token_fused


def score_variant(r: RunResult, baseline_raw_lp: float) -> float:
    lp_penalty = max(0.0, (baseline_raw_lp - r.mean_raw_logprob))
    kl_penalty = 0.01 * r.kl_cd_millibits
    return (
        -2.0 * lp_penalty
        - kl_penalty
        + 0.02 * r.unique_h
        + 0.5 * r.h_entropy
    )


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print_header()

    atlas_dir = Path("data/atlas")
    MODEL_DIR = Path("data/models/Bolmo-1B")
    prompt = "Language modeling is "

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model not found: {MODEL_DIR}")
    if not atlas_dir.exists():
        raise RuntimeError(f"Atlas not found: {atlas_dir}")

    print(f"Model: {MODEL_DIR}")
    print(f"Atlas: {atlas_dir}")

    device = detect_device()
    model, tokenizer = load_bolmo(MODEL_DIR, device)
    print(f"vocab_size: {model.config.vocab_size}")

    print("Physics tables...")
    tables = build_physics_tables()

    token_offset = int(getattr(tokenizer, "offset", 4))
    token_byte, token_fused = build_token_maps(int(model.config.vocab_size), token_offset)

    calib = run_calibration(model, tokenizer, atlas_dir, tables, prompt=prompt, steps=256)
    print(f"Calibration summary: direct R²={calib.r2_direct:.4f}, complement R²={calib.r2_complement:.4f}, prefer={'COMP' if calib.use_complement else 'DIR'}")

    cfg = CouplingConfig(
        routed_layers=[],
        store_layer_telemetry=False,
        couple_local_decoder=True,
        couple_local_encoder=False,
    )
    labe = GyroLabe(model, atlas_dir=atlas_dir, config=cfg, token_offset=token_offset)
    labe.install()

    results: list[RunResult] = []

    def run_variant(name: str, mods: LogitMods, use_mask: bool, proj: Projector | None, n: int, text_chars: int) -> RunResult:
        rr = bolmo_generate_one(
            model=model,
            tokenizer=tokenizer,
            atlas_dir=atlas_dir,
            prompt=prompt,
            max_new_tokens=n,
            labe=(labe if use_mask else None),
            mods=mods,
            tables=tables,
            token_byte=token_byte,
            token_fused=token_fused,
            projector=proj,
        )
        rr.name = name
        print_run_result(rr, text_chars=text_chars)
        return rr

    print("Baselines...")
    mods_base = LogitMods(temperature=0.7, top_k=40, asi_core=False)

    r0 = run_variant("B0_uncoupled", mods_base, use_mask=False, proj=None, n=512, text_chars=800)
    results.append(r0)

    r1 = run_variant("B1_localmask", mods_base, use_mask=True, proj=None, n=512, text_chars=800)
    results.append(r1)
    baseline_lp = r1.mean_raw_logprob

    best_layer = calib.runtime_best_layer
    fwd = calib.hid_to_bulk
    back = calib.bulk_to_hid

    # projector alpha reduced (prevent shattering)
    proj_bulk = Projector(layer_idx=best_layer, fwd=fwd, back=back, alpha=0.02, erase_bits=None, use_complement=calib.use_complement)
    proj_gauge = Projector(layer_idx=best_layer, fwd=fwd, back=back, alpha=0.02, erase_bits=[8, 9, 10, 11, 20, 21, 22, 23], use_complement=calib.use_complement)

    print("ASI Architecture Suite...")
    mods_asi = LogitMods(
        temperature=0.7,
        top_k=40,
        asi_core=True,
        orbit_reward=0.5,        # Reduced from 1.5
        orbit_break_penalty=0.1, # Reduced from 0.5
        friction_scale=0.01,     # Reduced from 0.05
        fuse_reward=0.3,         # Reduced from 1.0
        fuse_forbid=False,       # CRITICAL: Allow fused tokens to exist naturally
    )

    results.append(run_variant("V_ASI_Core+M", mods_asi, use_mask=True, proj=None, n=512, text_chars=800))
    results.append(run_variant("V_Proj_Bulk+M", mods_base, use_mask=True, proj=proj_bulk, n=512, text_chars=800))
    results.append(run_variant("V_Proj_Gauge+M", mods_base, use_mask=True, proj=proj_gauge, n=512, text_chars=800))
    results.append(run_variant("V_ASI_Full+M", mods_asi, use_mask=True, proj=proj_bulk, n=512, text_chars=800))

    print("Long runs...")
    results.append(run_variant("B0_LONG", mods_base, use_mask=False, proj=None, n=2048, text_chars=500))
    results.append(run_variant("B1_LONG", mods_base, use_mask=True, proj=None, n=2048, text_chars=500))
    results.append(run_variant("V_ASI_Core+M_LONG", mods_asi, use_mask=True, proj=None, n=2048, text_chars=500))
    results.append(run_variant("V_ASI_Full+M_LONG", mods_asi, use_mask=True, proj=proj_bulk, n=2048, text_chars=500))

    # Separator line #2 and #3 (keep total <= 5)
    print("=" * 72)
    print("FINAL SUMMARY TABLE")
    print("name               bytes   tok/s   raw_lp    KL(mb)   horizons   H(h)    cd   fused%    score")
    for r in results:
        sc = score_variant(r, baseline_lp)
        print(
            f"{r.name:18s}  {r.n_bytes:5d}  {r.tok_s:6.2f}  {r.mean_raw_logprob:7.4f}  "
            f"{r.kl_cd_millibits:7.2f}  {r.unique_h:3d}/256  {r.h_entropy:5.2f}  "
            f"{r.mean_code_dist:4.2f}  {100*r.fused_rate:6.1f}  {sc:7.3f}"
        )
    print("=" * 72)
    print(f"Calibration: direct R²={calib.r2_direct:.4f}, complement R²={calib.r2_complement:.4f}, prefer={'COMP' if calib.use_complement else 'DIR'}")


if __name__ == "__main__":
    main()