#!/usr/bin/env python3
# scripts/run_gyrolabe_experiments.py
"""
Focused GyroLabe x Bolmo observatory.

One script, one run.

Purpose
1) Confirm kernel geometry is present in Bolmo byte embeddings using the correct probe (linear mask decoding).
2) Decode boundary predictor physics by measuring real boundary events during decoding.
   This requires instrumenting the internal sampling decision: whether the sampled token was boundary-fused.
3) Relate boundary events to kernel-native observables (code distance, vertex and phase changes, horizon distance).
4) Compare uncoupled vs local-decoder coupled runs without broad ablation matrices.
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings(
    "ignore", message=".*rope_config_validation.*",
    category=FutureWarning, module="transformers.*",
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.constants import (
    LAYER_MASK_12,
    mask12_for_byte,
    popcount,
    trajectory_parity_commitment,
    vertex_charge_from_mask,
)
from src.router.kernel import RouterKernel
from src.tools.gyrolabe import (
    CouplingConfig,
    GyroLabe,
    detect_device,
    get_code_distance_matrix,
    get_code_prior,
    get_mask12_table,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_DIR = Path("data/models/Bolmo-1B")
ATLAS_DIR = Path("data/atlas")

PROMPT = "Language modeling is "
PROMPT_LONG = (
    "The relationship between structure and meaning in natural language reveals deep principles. "
    "We study how discrete systems preserve coherence under transformation. "
)

TOKENS_TRACE = 1024
TEMPERATURE = 0.7
TOP_K = 40
SEED = 42

BOLMO_TOKEN_OFFSET = 4
BOLMO_BOUNDARY_OFFSET = BOLMO_TOKEN_OFFSET + 256


# =====================================================================
# Bolmo loading and patches
# =====================================================================

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
    """Ensure next run starts in prefill mode for the local encoder/decoder."""
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


# =====================================================================
# Utility
# =====================================================================

def entropy_from_counts(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def point_biserial(binary: np.ndarray, continuous: np.ndarray) -> float:
    b = np.asarray(binary, dtype=bool)
    x = np.asarray(continuous, dtype=np.float64)
    if b.all() or (~b).all():
        return 0.0
    s = x.std()
    if s < 1e-12:
        return 0.0
    m1, m0 = x[b].mean(), x[~b].mean()
    n1, n0, n = b.sum(), (~b).sum(), len(x)
    return float((m1 - m0) * math.sqrt(n1 * n0 / (n * n)) / s)


def extract_byte_from_token(token_id: int) -> int:
    if token_id >= BOLMO_BOUNDARY_OFFSET:
        return token_id - BOLMO_BOUNDARY_OFFSET
    if token_id >= BOLMO_TOKEN_OFFSET:
        return token_id - BOLMO_TOKEN_OFFSET
    return token_id & 0xFF


def ridge_r2_dual(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> float:
    """Ridge regression R² using dual form: W = Xᵀ(XXᵀ + αI)⁻¹Y. Good for n << d."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    A = Xc @ Xc.T + alpha * np.eye(n)
    Yhat = Xc @ (Xc.T @ np.linalg.solve(A, Yc))
    ss_res = float(np.sum((Yc - Yhat) ** 2))
    ss_tot = float(np.sum(Yc ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def topk_sample(logits: torch.Tensor, temperature: float, top_k: int) -> tuple[int, torch.Tensor]:
    """Sample a token and return (token_id, full_softmax_probs)."""
    scaled = logits / max(temperature, 1e-8)
    if top_k > 0:
        k = min(int(top_k), int(scaled.numel()))
        topv, topi = torch.topk(scaled, k=k)
        probs = torch.softmax(topv, dim=-1)
        j = int(torch.multinomial(probs, 1).item())
        tok = int(topi[j].item())
    else:
        tok = int(torch.multinomial(torch.softmax(scaled, dim=-1), 1).item())
    full = torch.softmax(scaled, dim=-1)
    return tok, full


def patch_lengths(flags: np.ndarray) -> np.ndarray:
    """Compute patch lengths from a boolean boundary flag array."""
    lens: list[int] = []
    cur = 0
    for x in flags:
        cur += 1
        if x:
            lens.append(cur)
            cur = 0
    if cur > 0:
        lens.append(cur)
    return np.array(lens, dtype=np.int32)


# =====================================================================
# Boundary predictor access (prefill)
# =====================================================================

def compute_prefill_boundaries(model: Any, input_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Returns (boundary_logprobs, boundary_mask) for prefill, both 1D length L."""
    bolmo_reset_local_caches(model)

    from utils_bolmo import MaskState  # type: ignore

    device = input_ids.device
    tok = model.model.tokenizer

    if model.model.local_encoder.add_expanded_embeddings:
        expanded = torch.tensor(
            tok.expand_byte_ids(input_ids[0].tolist()), device=device, dtype=torch.long,
        ).unsqueeze(0)
    else:
        expanded = None

    seq_start = (input_ids == tok.pad_token_id).sum(-1)

    _, _, boundary_logprobs, boundary_mask = model.model.local_encoder.forward(
        input_ids,
        expanded_input_ids=expanded,
        boundary_state=MaskState(torch.full((1,), fill_value=False, device=device, dtype=torch.bool)),
        pad_state=MaskState(torch.zeros((1,), device=device, dtype=torch.bool)),
        sequence_start_indices=seq_start,
    )

    if boundary_logprobs is None or boundary_mask is None:
        raise RuntimeError(
            "local_encoder.forward returned None for boundary prediction. "
            "Ensure free_inference_cache() was called before prefill."
        )

    return (
        boundary_logprobs[0].detach().float().cpu().numpy(),
        boundary_mask[0].detach().cpu().numpy().astype(bool),
    )


# =====================================================================
# Traced decoding (batch_size=1)
# =====================================================================

@dataclass
class TraceResult:
    text: str
    elapsed: float
    n_tokens: int
    gen_token_ids_raw: list[int]
    gen_bytes: list[int]
    boundary_fused: np.ndarray       # bool per generated step
    boundary_mass: np.ndarray        # float per step
    boundary_delta: np.ndarray       # float per step: logit(fused_b) - logit(plain_b)
    kernel_h: np.ndarray             # horizon after step
    kernel_chi: np.ndarray           # vertex after step
    kernel_p: np.ndarray             # phase after step
    kernel_hd: np.ndarray            # horizon_distance after step
    kernel_cd: np.ndarray            # code distance between consecutive horizons
    byte_weight: np.ndarray          # popcount(mask12(byte))


@torch.inference_mode()
def bolmo_generate_trace(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    labe: GyroLabe | None,
) -> TraceResult:
    """
    Focused Bolmo generation for batch_size=1 exposing:
    - whether the sampled token was boundary-fused (tok >= boundary_offset)
    - probability mass on boundary-fused outputs
    - fused vs nonfused logit delta for the sampled byte
    - kernel-native observables per emitted byte
    """
    from utils_bolmo import MaskState  # type: ignore

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(model.parameters()).device
    tok = tokenizer

    enc = tok(prompt, return_tensors="pt")
    input_ids_full = enc.input_ids.to(device)
    prompt_ids = input_ids_full[0].tolist()

    bolmo_reset_local_caches(model)

    boundary_offset = tok.offset + 256
    eos = tok.eos_token_id
    bpe_end = tok.bpe_token_end_id

    # Prime GyroLabe kernel from prompt
    if labe is not None:
        labe.reset()
        labe.prime_from_tokens(prompt_ids)

    # Build expanded input ids for prefill
    expand_input_ids = model.model.local_encoder.add_expanded_embeddings
    if expand_input_ids:
        expanded_full = torch.tensor(
            tok.expand_byte_ids(prompt_ids), device=device, dtype=torch.long,
        ).unsqueeze(0)
    else:
        expanded_full = None

    # Prefill boundary prediction
    boundary_mask_full = model.model.prefill_boundary_prediction_forward(
        input_ids_full,
        expanded_input_ids=expanded_full,
        sequence_start_indices=(input_ids_full == tok.pad_token_id).sum(-1),
    )

    if boundary_mask_full is None:
        raise RuntimeError(
            "prefill_boundary_prediction_forward returned None. "
            "Ensure free_inference_cache() is called before prefill."
        )

    # Prepare caches
    model.model.local_encoder.free_inference_cache()
    model.model.local_decoder.free_inference_cache()
    model.model.local_encoder.prepare_inference_cache(1)
    model.model.local_decoder.prepare_inference_cache(1)

    # Roll back by one to account for lookahead
    boundary_mask = boundary_mask_full[:, :-1]
    forced_decoding_id = int(input_ids_full[0, -1].item())

    input_ids = input_ids_full[:, :-1]
    expanded = expanded_full[:, :-1] if expanded_full is not None else None

    seq_start = (input_ids == tok.pad_token_id).sum(-1)

    # Generation state
    generated = input_ids.clone()
    non_boundary_bytes: list[int] = [int(input_ids_full[0, -1].item())]
    gen_token_ids_raw: list[int] = []

    boundary_state = MaskState(boundary_mask[:, -1].clone())
    pad_state = MaskState(torch.zeros((1,), dtype=torch.bool, device=device))

    global_past = None
    is_first = True

    # Independent tracking kernel
    kernel = RouterKernel(atlas_dir=ATLAS_DIR)
    for tid in prompt_ids:
        kernel.step_byte(extract_byte_from_token(tid))
    cdm = get_code_distance_matrix()

    # Accumulators
    kernel_h: list[int] = []
    kernel_chi: list[int] = []
    kernel_p: list[int] = []
    kernel_hd: list[int] = []
    kernel_cd: list[int] = []
    byte_w: list[int] = []
    boundary_fused_list: list[bool] = []
    boundary_mass_list: list[float] = []
    boundary_delta_list: list[float] = []

    last_h = int(kernel.current_horizon.item())

    t0 = time.perf_counter()

    while len(gen_token_ids_raw) < max_new_tokens:
        if labe is not None:
            labe.begin_step()

        # Model input: full sequence on first step, single token thereafter
        if is_first:
            input_ids_for_model = generated
            expanded_for_model = expanded
        else:
            last_tok = int(non_boundary_bytes[-1])
            input_ids_for_model = torch.tensor([[last_tok]], device=device, dtype=generated.dtype)
            if expand_input_ids:
                expanded_last = tok.expand_byte_ids(generated[0].tolist(), n_last=1)[-1]
                expanded_for_model = torch.tensor([[expanded_last]], device=device, dtype=torch.long)
            else:
                expanded_for_model = None

        out = model.forward(
            input_ids_for_model,
            expanded_input_ids=expanded_for_model,
            boundary_mask=boundary_mask if is_first else None,
            boundary_state=boundary_state,
            pad_state=pad_state,
            sequence_start_indices=seq_start,
            logits_to_keep=1,
            use_cache=True,
            past_key_values=global_past,
        )
        global_past = out.past_key_values

        if labe is not None:
            labe.end_step()

        logits = out.logits[0, -1].float()

        # Forced decode on first step to preserve lookahead alignment
        if forced_decoding_id is not None:
            no_bnd = forced_decoding_id
            yes_bnd = forced_decoding_id + boundary_offset
            forced_logits = torch.full_like(logits, -1e9)
            forced_logits[no_bnd] = logits[no_bnd]
            if 0 <= yes_bnd < forced_logits.numel():
                forced_logits[yes_bnd] = logits[yes_bnd]
            logits = forced_logits
            forced_decoding_id = None

        # BU-Ingress adjustment
        if labe is not None and top_k > 0:
            scaled = logits / max(temperature, 1e-8)
            k = min(64, int(scaled.numel()))
            topv, topi = torch.topk(scaled, k=k)
            topv_adj = labe.adjust_logits_bu_ingress(topi, topv)
            logits = logits.clone()
            logits[topi] = topv_adj * max(temperature, 1e-8)

        tok_raw, probs_full = topk_sample(logits, temperature=temperature, top_k=top_k)
        gen_token_ids_raw.append(tok_raw)

        # Boundary classification
        is_fused = tok_raw >= boundary_offset
        boundary_fused_list.append(is_fused)

        if boundary_offset + 256 <= int(probs_full.numel()):
            b_mass = float(probs_full[boundary_offset:boundary_offset + 256].sum().item())
        else:
            b_mass = 0.0
        boundary_mass_list.append(b_mass)

        # Fused vs nonfused logit delta
        tok_unfused = tok_raw - boundary_offset if is_fused else tok_raw
        delta = 0.0
        if BOLMO_TOKEN_OFFSET <= tok_unfused < BOLMO_TOKEN_OFFSET + 256:
            b = tok_unfused - BOLMO_TOKEN_OFFSET
            t_plain = BOLMO_TOKEN_OFFSET + b
            t_fused = BOLMO_TOKEN_OFFSET + 256 + b
            if 0 <= t_plain < logits.numel() and 0 <= t_fused < logits.numel():
                delta = float((logits[t_fused] - logits[t_plain]).item())
        boundary_delta_list.append(delta)

        # Append to internal stream
        generated = torch.cat(
            [generated, torch.tensor([[tok_raw]], device=device, dtype=generated.dtype)], dim=1,
        )

        # Emit byte (skip bpe_end)
        if tok_unfused != bpe_end:
            non_boundary_bytes.append(int(tok_unfused))

            if labe is not None:
                labe.advance_with_token(int(tok_unfused))

            b_drive = extract_byte_from_token(int(tok_unfused))
            kernel.step_byte(b_drive)

            h = int(kernel.current_horizon.item())
            chi = int(kernel.current_vertex.item())
            p = int(kernel.current_phase.item())

            sig = kernel.signature()
            a12 = int(sig.a_hex, 16)
            b12 = int(sig.b_hex, 16)
            hd = popcount(a12 ^ (b12 ^ LAYER_MASK_12))

            kernel_h.append(h)
            kernel_chi.append(chi)
            kernel_p.append(p)
            kernel_hd.append(hd)
            kernel_cd.append(int(cdm[last_h, h]))
            byte_w.append(popcount(mask12_for_byte(b_drive)))
            last_h = h

        # Update boundary and pad states
        next_is_boundary = (tok_raw == bpe_end) or is_fused
        boundary_state = MaskState(torch.tensor([next_is_boundary], device=device, dtype=torch.bool))
        pad_state = MaskState(torch.tensor([tok_raw == bpe_end], device=device, dtype=torch.bool))

        is_first = False

        if tok_raw == eos or tok_raw == eos + boundary_offset:
            break

    elapsed = time.perf_counter() - t0

    # Decode text
    text = tok.decode([int(t) for t in non_boundary_bytes], skip_special_tokens=True)
    gen_bytes_kernel = [extract_byte_from_token(t) for t in non_boundary_bytes[1:]]

    bolmo_reset_local_caches(model)

    return TraceResult(
        text=text,
        elapsed=elapsed,
        n_tokens=len(gen_token_ids_raw),
        gen_token_ids_raw=gen_token_ids_raw,
        gen_bytes=gen_bytes_kernel,
        boundary_fused=np.array(boundary_fused_list, dtype=bool),
        boundary_mass=np.array(boundary_mass_list, dtype=np.float32),
        boundary_delta=np.array(boundary_delta_list, dtype=np.float32),
        kernel_h=np.array(kernel_h, dtype=np.int32),
        kernel_chi=np.array(kernel_chi, dtype=np.int32),
        kernel_p=np.array(kernel_p, dtype=np.int32),
        kernel_hd=np.array(kernel_hd, dtype=np.int32),
        kernel_cd=np.array(kernel_cd, dtype=np.int32),
        byte_weight=np.array(byte_w, dtype=np.int32),
    )


# =====================================================================
# Analysis blocks
# =====================================================================

def print_kv(k: str, v: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{k}: {v}")


def print_patch_stats(label: str, flags: np.ndarray) -> None:
    """Print patch length statistics from a boolean flag array."""
    pl = patch_lengths(flags)
    if pl.size == 0:
        return
    print_kv(f"{label} patch count", int(len(pl)))
    print_kv(f"{label} patch len mean", f"{pl.mean():.2f}")
    print_kv(f"{label} patch len median", int(np.median(pl)))
    print_kv(f"{label} patch len max", int(pl.max()))


def block_arch(model: Any) -> None:
    cfg = model.config
    print("Block 1: architecture")
    for name in [
        "hidden_size", "intermediate_size", "local_intermediate_size",
        "num_hidden_layers", "num_local_encoder_layers", "num_local_decoder_layers",
        "max_position_embeddings", "rope_theta", "vocab_size",
    ]:
        print_kv(name, getattr(cfg, name, "N/A"))
    print_kv("boundary_lookahead", cfg.boundary_predictor_lookahead)
    print("")


def block_static_geometry(model: Any) -> None:
    print("Block 2: static geometry probes (embeddings)")
    emb_weight = model.model.local_encoder.byte_embedding.weight.detach().float().cpu()
    byte_embs = emb_weight[BOLMO_TOKEN_OFFSET:BOLMO_TOKEN_OFFSET + 256].numpy()

    mask_table = get_mask12_table()

    # Mask bits target matrix [256, 12]
    mask_bits = np.zeros((256, 12), dtype=np.float64)
    for b in range(256):
        m = int(mask_table[b])
        for bit in range(12):
            mask_bits[b, bit] = 1.0 if ((m >> bit) & 1) else -1.0
    r2_mask = ridge_r2_dual(byte_embs, mask_bits, alpha=1.0)

    # Vertex one-hot target matrix [256, 4]
    vertex_onehot = np.zeros((256, 4), dtype=np.float64)
    for b in range(256):
        vertex_onehot[b, vertex_charge_from_mask(int(mask_table[b]))] = 1.0
    r2_vertex = ridge_r2_dual(byte_embs, vertex_onehot, alpha=1.0)

    # Family clustering ratio
    norms = np.linalg.norm(byte_embs, axis=1, keepdims=True) + 1e-12
    cos_dist = 1.0 - ((byte_embs / norms) @ (byte_embs / norms).T)
    families = np.array([(b ^ 0xAA) >> 6 & 0x3 for b in range(256)])
    within, between = [], []
    for i in range(256):
        for j in range(i + 1, 256):
            d = cos_dist[i, j]
            if families[i] == families[j]:
                within.append(d)
            else:
                between.append(d)
    between_mean = float(np.mean(between)) if len(between) > 0 else 1e-12
    ratio = float(np.mean(within) / max(between_mean, 1e-12))

    print_kv("R² embeddings -> mask12 (dual ridge)", f"{r2_mask:.6f}")
    print_kv("R² embeddings -> vertex (dual ridge)", f"{r2_vertex:.6f}")
    print_kv("family within/between cosine-dist ratio", f"{ratio:.4f}")
    print("")


def block_prefill_boundary(model: Any, tokenizer: Any, prompt: str) -> None:
    print("Block 3: prefill boundary predictor (non-causal, lookahead=1)")
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    blogp, bmask = compute_prefill_boundaries(model, input_ids)
    L = len(bmask)
    eff = max(0, L - 1)
    bmask_eff = bmask[:eff]

    rate = float(bmask_eff.mean()) if eff > 0 else 0.0
    print_kv("prompt length (bytes)", L)
    print_kv("boundary rate (prefill)", f"{rate * 100:.2f}%")

    # Correlate with kernel mask geometry on prompt byte transitions
    prompt_ids = enc.input_ids[0].tolist()
    bytes_prompt = [extract_byte_from_token(t) for t in prompt_ids]
    mask_table = get_mask12_table()

    if eff > 5:
        d_code = np.array([
            popcount(int(mask_table[bytes_prompt[t]]) ^ int(mask_table[bytes_prompt[t + 1]]))
            for t in range(eff)
        ], dtype=np.float64)
        d_vertex = np.array([
            int(vertex_charge_from_mask(int(mask_table[bytes_prompt[t]])) !=
                vertex_charge_from_mask(int(mask_table[bytes_prompt[t + 1]])))
            for t in range(eff)
        ], dtype=np.float64)
        d_weight = np.array([
            popcount(int(mask_table[bytes_prompt[t]])) + popcount(int(mask_table[bytes_prompt[t + 1]]))
            for t in range(eff)
        ], dtype=np.float64)

        if bmask_eff.any() and (~bmask_eff).any():
            print_kv("r_pb(boundary, code_dist)", f"{point_biserial(bmask_eff, d_code):.4f}")
            print_kv("r_pb(boundary, vertex_change)", f"{point_biserial(bmask_eff, d_vertex):.4f}")
            print_kv("r_pb(boundary, weight_sum)", f"{point_biserial(bmask_eff, d_weight):.4f}")

    print_patch_stats("prefill", bmask_eff)
    print("")


def summarize_trace(name: str, tr: TraceResult) -> None:
    print(f"Block 4: traced decoding run ({name})")
    print_kv("tokens generated (steps)", tr.n_tokens)
    print_kv("elapsed", f"{tr.elapsed:.1f}s")
    print_kv("tok/s", f"{tr.n_tokens / max(tr.elapsed, 1e-9):.2f}")
    print_kv("text preview", tr.text[:160].replace("\n", " ") + "...")
    print("")

    if tr.kernel_cd.size > 0:
        P_code, _ = get_code_prior()
        emp_cd = np.bincount(tr.kernel_cd, minlength=13).astype(np.float64)
        emp_cd /= max(emp_cd.sum(), 1)
        kl = kl_bits(emp_cd, P_code[:13])

        print_kv("kernel code_dist mean", f"{tr.kernel_cd.mean():.2f}")
        print_kv("kernel code_dist std", f"{tr.kernel_cd.std():.2f}")
        print_kv("KL(empirical cd || P_code)", f"{kl * 1000:.2f} millibits")

        h_counts = np.bincount(tr.kernel_h.astype(np.int64), minlength=256)
        print_kv("unique horizons", int((h_counts > 0).sum()))
        print_kv("H(h)", f"{entropy_from_counts(h_counts):.2f} bits")

    if tr.gen_bytes:
        O, E, parity = trajectory_parity_commitment(tr.gen_bytes)
        print_kv("parity O", f"0x{O:03x}")
        print_kv("parity E", f"0x{E:03x}")
        print_kv("parity n%2", parity)

    print("")


def analyze_decode_boundaries(name: str, tr: TraceResult) -> None:
    print(f"Block 5: decoding boundary physics ({name})")

    n_emit = int(tr.kernel_cd.size)
    if n_emit == 0:
        print_kv("status", "no data")
        print("")
        return

    # Align boundary flags to emitted bytes
    bf = tr.boundary_fused[:n_emit]
    bm = tr.boundary_mass[:n_emit]
    bd = tr.boundary_delta[:n_emit]

    rate = float(bf.mean())
    print_kv("boundary fused rate", f"{rate * 100:.3f}%")
    print_kv("boundary mass mean", f"{bm.mean():.4f}")
    print_kv("boundary mass std", f"{bm.std():.4f}")
    print_kv("boundary delta mean (fused-plain logit)", f"{bd.mean():.4f}")

    # Kernel-linked observables
    cd = tr.kernel_cd.astype(np.float64)
    hd = tr.kernel_hd.astype(np.float64)
    chi = tr.kernel_chi.astype(np.int64)
    p = tr.kernel_p.astype(np.int64)
    w = tr.byte_weight.astype(np.float64)

    chi_chg = np.zeros_like(chi, dtype=bool)
    p_chg = np.zeros_like(p, dtype=bool)
    chi_chg[1:] = chi[1:] != chi[:-1]
    p_chg[1:] = p[1:] != p[:-1]

    if bf.any() and (~bf).any():
        print_kv("r_pb(boundary, code_dist)", f"{point_biserial(bf, cd):.4f}")
        print_kv("r_pb(boundary, horizon_distance)", f"{point_biserial(bf, hd):.4f}")
        print_kv("r_pb(boundary, vertex_changed)", f"{point_biserial(bf, chi_chg.astype(np.float64)):.4f}")
        print_kv("r_pb(boundary, phase_changed)", f"{point_biserial(bf, p_chg.astype(np.float64)):.4f}")
        print_kv("r_pb(boundary, byte_mask_weight)", f"{point_biserial(bf, w):.4f}")
    else:
        print_kv("note", "boundary events are all one class (rate ~0 or ~1)")

    # Continuous correlations with boundary mass
    if bm.std() > 1e-9:
        r_cd = float(np.corrcoef(bm, cd)[0, 1]) if cd.std() > 1e-9 else 0.0
        r_hd = float(np.corrcoef(bm, hd)[0, 1]) if hd.std() > 1e-9 else 0.0
        print_kv("corr(boundary_mass, code_dist)", f"{r_cd:.4f}")
        print_kv("corr(boundary_mass, horizon_distance)", f"{r_hd:.4f}")

    print_patch_stats("decode, fused", bf)
    print("")


def block_localdec_layers(labe: GyroLabe) -> None:
    print("Block 6: local decoder per-layer correlation (from GyroLabe telemetry)")
    if not labe.trajectory:
        print_kv("status", "no trajectory")
        print("")
        return

    per_layer: dict[int, list[float]] = {1000: [], 1001: [], 1002: [], 1003: []}
    for step in labe.trajectory:
        for d in step.get("layers", []):
            lid = int(d["layer_idx"])
            if lid in per_layer:
                per_layer[lid].append(float(d.get("correlation", 0.0)))

    for lid in sorted(per_layer.keys()):
        arr = np.array(per_layer[lid], dtype=np.float64)
        if arr.size == 0:
            print_kv(f"layer {lid}", "no data")
        else:
            print_kv(f"layer {lid} corr mean", f"{arr.mean():.4f}")
            print_kv(f"layer {lid} corr std", f"{arr.std():.4f}")
    print("")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("GyroLabe focused observatory (Bolmo)")
    print_kv("model", str(MODEL_DIR))
    print_kv("atlas", str(ATLAS_DIR))
    print_kv("prompt", repr(PROMPT))
    print_kv("tokens_trace", TOKENS_TRACE)
    print("")

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model not found: {MODEL_DIR}")
    if not ATLAS_DIR.exists():
        raise RuntimeError(f"Atlas not found: {ATLAS_DIR}")

    device = detect_device()
    model, tokenizer = load_bolmo(MODEL_DIR, device)

    # Blocks 1-2: no generation required
    block_arch(model)
    block_static_geometry(model)

    # Block 3: prefill boundary predictor on longer prompt
    block_prefill_boundary(model, tokenizer, PROMPT_LONG)

    # Run A: uncoupled traced decoding
    tr_unc = bolmo_generate_trace(
        model=model, tokenizer=tokenizer, prompt=PROMPT_LONG,
        max_new_tokens=TOKENS_TRACE, temperature=TEMPERATURE,
        top_k=TOP_K, seed=SEED, labe=None,
    )
    summarize_trace("uncoupled", tr_unc)
    analyze_decode_boundaries("uncoupled", tr_unc)

    # Run B: local decoder coupled traced decoding
    cfg = CouplingConfig(
        couple_local_decoder=True,
        couple_local_encoder=False,
        store_layer_telemetry=True,
    )
    labe = GyroLabe(
        model=model, atlas_dir=ATLAS_DIR,
        config=cfg, token_offset=BOLMO_TOKEN_OFFSET,
    )
    labe.install()

    try:
        tr_c = bolmo_generate_trace(
            model=model, tokenizer=tokenizer, prompt=PROMPT_LONG,
            max_new_tokens=TOKENS_TRACE, temperature=TEMPERATURE,
            top_k=TOP_K, seed=SEED + 1, labe=labe,
        )
        summarize_trace("local_dec coupled", tr_c)
        analyze_decode_boundaries("local_dec coupled", tr_c)

        # GyroLabe coupling summary
        s = labe.stats()
        print("Block 6b: GyroLabe coupling summary (local_dec)")
        print_kv("mean correlation", s.get("mean_correlation", 0.0))
        print_kv("std correlation", s.get("std_correlation", 0.0))
        print_kv("mean code_dist (telemetry)", s.get("mean_code_dist", 0.0))
        print_kv("unique_h (telemetry)", f"{s.get('unique_h', 0)}/256")
        print_kv("h_entropy (telemetry)", s.get("h_entropy", 0.0))
        print_kv("chi_dist (telemetry)", s.get("chi_dist", []))
        print("")

        block_localdec_layers(labe)
    finally:
        labe.restore()

    print("Done.")


if __name__ == "__main__":
    main()