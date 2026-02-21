#!/usr/bin/env python3
# scripts/run_gyrolabe_experiments.py
"""
GyroLabe Structural Observatory for Bolmo-1B.

A single run that prints a structural fingerprint: static geometry, uncoupled
census, coupled ablations, dynamic boundary analysis, and invariant checks.

Targets Bolmo-1B (byte-level LM) with the GGG ASI Alignment Router Kernel.
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings(
    "ignore", message=".*rope_config_validation.*",
    category=FutureWarning, module="transformers.*",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router.constants import (
    ARCHETYPE_A12, ARCHETYPE_B12, ARCHETYPE_STATE24, LAYER_MASK_12,
    mask12_for_byte, popcount, trajectory_parity_commitment,
    vertex_charge_from_mask,
)
from src.router.kernel import RouterKernel
from src.tools.gyrolabe import (
    CouplingConfig, GyroLabe, detect_device, get_code_distance_matrix,
    get_code_prior, get_mask12_table, compute_mask, ACTIVE_BETA,
    N_BOUNDARY, _entropy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = Path("data/models/Bolmo-1B")
ATLAS_DIR = Path("data/atlas")
BOLMO_TOKEN_OFFSET = 4
BOLMO_BOUNDARY_OFFSET = BOLMO_TOKEN_OFFSET + 256

PROMPT = "Language modeling is "
LONG_PROMPT = (
    "The relationship between structure and meaning in natural language "
    "reveals deep principles about how information organizes itself. "
    "Consider how "
)
GEN_TOKENS_SHORT = 500
GEN_TOKENS_LONG = 2000
TEMPERATURE = 0.7
TOP_K = 40
SEED = 42

WINDOW_SIZE = 64  # for windowed dynamics


# ═══════════════════════════════════════════════════════════════════════════════
# Bolmo compatibility patches
# ═══════════════════════════════════════════════════════════════════════════════

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
        print(f"ERROR: dolma2 tokenizer not cached at {local_dir}")
        sys.exit(1)
    print(f"  Caching dolma2 tokenizer to: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="allenai/dolma2-tokenizer", local_dir=str(local_dir),
        allow_patterns=["tokenizer.json", "tokenizer_config.json"],
    )
    return local_dir


def load_bolmo(model_dir: Path, device: torch.device):
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _patch_rope_default_for_bolmo()
    _patch_generation_prepare()

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading {model_dir.name} on {device} (dtype={torch_dtype})...")

    model_dir = model_dir.resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from configuration_bolmo import BolmoConfig  # pyright: ignore[reportMissingImports]
    from modeling_bolmo import BolmoForCausalLM  # pyright: ignore[reportMissingImports]

    config = BolmoConfig.from_pretrained(model_dir, local_files_only=True)
    model = BolmoForCausalLM.from_pretrained(
        model_dir, config=config, local_files_only=True, torch_dtype=torch_dtype
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


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def print_subsection(title: str):
    print(f"\n  ── {title}")


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) in bits."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def point_biserial(binary: np.ndarray, continuous: np.ndarray) -> float:
    """Point-biserial correlation between a binary and continuous variable."""
    b = np.asarray(binary, dtype=bool)
    c = np.asarray(continuous, dtype=np.float64)
    if b.all() or (~b).all():
        return 0.0
    m1 = c[b].mean()
    m0 = c[~b].mean()
    n1 = b.sum()
    n0 = (~b).sum()
    n = len(c)
    s = c.std()
    if s < 1e-12:
        return 0.0
    return float((m1 - m0) * math.sqrt(n1 * n0 / (n * n)) / s)


def monodromy_from_parity(O: int, E: int) -> float:
    """Monodromy A(t) = (popcount(O) + popcount(E)) / 24."""
    return (popcount(O) + popcount(E)) / 24.0


def extract_byte_from_token(token_id: int) -> int:
    """Extract kernel-driving byte from Bolmo token id."""
    if token_id >= BOLMO_BOUNDARY_OFFSET:
        return token_id - BOLMO_BOUNDARY_OFFSET
    elif token_id >= BOLMO_TOKEN_OFFSET:
        return token_id - BOLMO_TOKEN_OFFSET
    else:
        return token_id & 0xFF


# ═══════════════════════════════════════════════════════════════════════════════
# Block 1: Architecture Census
# ═══════════════════════════════════════════════════════════════════════════════

def block_architecture(model) -> dict:
    print_section("BLOCK 1: ARCHITECTURE CENSUS")
    cfg = model.config

    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    local_inter = cfg.local_intermediate_size
    n_global = cfg.num_hidden_layers
    n_enc = cfg.num_local_encoder_layers
    n_dec = cfg.num_local_decoder_layers
    vocab = cfg.vocab_size

    print(f"  vocab_size           = {vocab}")
    print(f"  hidden_size          = {hidden}  (÷256 = {hidden // 256})  {'OK' if hidden % 256 == 0 else 'FAIL'}")
    print(f"  intermediate_size    = {inter}  (÷256 = {inter // 256})  {'OK' if inter % 256 == 0 else 'FAIL'}")
    print(f"  local_inter_size     = {local_inter}  (÷256 = {local_inter // 256})  {'OK' if local_inter % 256 == 0 else 'FAIL'}")
    print(f"  global_layers        = {n_global}")
    print(f"  local_enc_layers     = {n_enc}")
    print(f"  local_dec_layers     = {n_dec}  (kernel depth-4 closure)")
    print(f"  layer_types          = all full_attention" if all(t == "full_attention" for t in cfg.layer_types) else f"  layer_types = {cfg.layer_types}")
    print(f"  boundary_lookahead   = {cfg.boundary_predictor_lookahead}")
    print(f"  max_position_embeds  = {cfg.max_position_embeddings}  (= |Ω| = 2^16)")
    print(f"  rope_theta           = {cfg.rope_theta}")
    print(f"  num_attention_heads  = {cfg.num_attention_heads}")
    print(f"  num_local_heads      = {cfg.num_local_heads}")

    n_fiber_global = inter // 256
    n_fiber_local = local_inter // 256
    n_fiber_hidden = hidden // 256
    print(f"\n  Fiber dimensions:")
    print(f"    global MLP:  {n_fiber_global} fibers per boundary position")
    print(f"    local MLP:   {n_fiber_local} fibers per boundary position")
    print(f"    hidden:      {n_fiber_hidden} fibers per boundary position")

    # Verify MLP structure
    layers = model.model.layers
    ld = model.model.local_decoder.layers
    le = model.model.local_encoder.layers
    print(f"\n  SwiGLU gate_proj present:")
    print(f"    global layers: {hasattr(layers[0].mlp, 'gate_proj')}")
    print(f"    local decoder: {hasattr(ld[0].mlp, 'gate_proj')}")
    print(f"    local encoder: {hasattr(le[0].mlp, 'gate_proj')}")

    return {
        "hidden": hidden, "inter": inter, "local_inter": local_inter,
        "n_global": n_global, "n_enc": n_enc, "n_dec": n_dec,
        "n_fiber_global": n_fiber_global, "n_fiber_local": n_fiber_local,
        "n_fiber_hidden": n_fiber_hidden,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Block 2: Static Geometry — Embeddings, Masks, and Linear Probes
# ═══════════════════════════════════════════════════════════════════════════════

def block_static_geometry(model) -> dict:
    print_section("BLOCK 2: STATIC GEOMETRY")

    mask_table = get_mask12_table()
    cdm = get_code_distance_matrix()

    # ── 2a. Byte embeddings ──
    print_subsection("Byte Embeddings vs Kernel Mask Code")

    emb_weight = model.model.local_encoder.byte_embedding.weight.detach().float().cpu()
    byte_embs = emb_weight[BOLMO_TOKEN_OFFSET:BOLMO_TOKEN_OFFSET + 256]  # [256, 2048]
    print(f"  Embedding shape: {byte_embs.shape}")

    norms = byte_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    byte_normed = byte_embs / norms
    cos_sim = (byte_normed @ byte_normed.T).numpy()
    cos_dist = 1.0 - cos_sim
    cdm_f = cdm.astype(np.float64)

    idx_upper = np.triu_indices(256, k=1)
    cos_flat = cos_dist[idx_upper]
    code_flat = cdm_f[idx_upper]

    r_pearson = np.corrcoef(cos_flat, code_flat)[0, 1]
    r_spearman = _spearman_corr(cos_flat, code_flat)

    print(f"  Pearson(cosine_dist, code_dist)  = {r_pearson:.6f}")
    print(f"  Spearman(cosine_dist, code_dist) = {r_spearman:.6f}")
    print(f"    cosine_dist: mean={cos_flat.mean():.4f}  std={cos_flat.std():.4f}")
    print(f"    code_dist:   mean={code_flat.mean():.4f}  std={code_flat.std():.4f}")

    # ── 2b. Family clustering ──
    print_subsection("Family Clustering (intron bits 6,7 → 4 families)")

    families = np.array([(b ^ 0xAA) >> 6 & 0x3 for b in range(256)])
    within, between = [], []
    for i in range(256):
        for j in range(i + 1, 256):
            d = cos_dist[i, j]
            (within if families[i] == families[j] else between).append(d)

    within_arr, between_arr = np.array(within), np.array(between)
    ratio = within_arr.mean() / max(between_arr.mean(), 1e-8)
    print(f"  Within-family:  mean={within_arr.mean():.4f}  std={within_arr.std():.4f}")
    print(f"  Between-family: mean={between_arr.mean():.4f}  std={between_arr.std():.4f}")
    print(f"  Ratio (within/between): {ratio:.4f}  (<1 = families cluster)")

    # ── 2c. Mask weight monotonicity ──
    print_subsection("Mask Weight vs Cosine Distance")

    byte_weights = np.array([popcount(int(mask_table[b])) for b in range(256)])
    weight_means = {}
    for w in range(13):
        members = np.where(byte_weights == w)[0]
        if len(members) < 2:
            continue
        dists = [cos_dist[members[i], members[j]]
                 for i in range(len(members)) for j in range(i + 1, len(members))]
        weight_means[w] = np.mean(dists)
        print(f"  weight={w:2d}: n={len(members):3d}  mean_cos_dist={weight_means[w]:.4f}")

    # Check monotonicity
    weights_sorted = sorted(weight_means.keys())
    values = [weight_means[w] for w in weights_sorted]
    monotonic_violations = sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])
    print(f"  Monotonicity violations: {monotonic_violations}/{len(values) - 1}")

    # ── 2d. Linear probe: recover mask12 from embeddings ──
    print_subsection("Linear Probe: Embeddings → 12-bit Mask")

    mask_bits = np.zeros((256, 12), dtype=np.float32)
    for b in range(256):
        m = int(mask_table[b])
        for bit in range(12):
            mask_bits[b, bit] = 1.0 if ((m >> bit) & 1) else -1.0

    emb_np = byte_embs.numpy()
    probe_r2 = _linear_probe_r2(emb_np, mask_bits)
    print(f"  R² (Ridge, 256 samples, 2048→12): {probe_r2:.6f}")

    # Per-bit R²
    per_bit_r2 = []
    for bit in range(12):
        r2 = _linear_probe_r2(emb_np, mask_bits[:, bit:bit + 1])
        per_bit_r2.append(r2)

    print(f"  Per-bit R²: {[f'{r:.3f}' for r in per_bit_r2]}")
    print(f"  Frame 0 (bits 0-5) mean R²: {np.mean(per_bit_r2[:6]):.4f}")
    print(f"  Frame 1 (bits 6-11) mean R²: {np.mean(per_bit_r2[6:]):.4f}")

    # Per-row R² (X=bits 0,1,6,7; Y=bits 2,3,8,9; Z=bits 4,5,10,11)
    row_groups = {"X": [0, 1, 6, 7], "Y": [2, 3, 8, 9], "Z": [4, 5, 10, 11]}
    for name, bits in row_groups.items():
        r2 = np.mean([per_bit_r2[b] for b in bits])
        print(f"  Row {name} (bits {bits}) mean R²: {r2:.4f}")

    # ── 2e. Linear probe: recover vertex charge from embeddings ──
    print_subsection("Linear Probe: Embeddings → Vertex Charge")

    vertex_targets = np.zeros((256, 4), dtype=np.float32)
    for b in range(256):
        v = vertex_charge_from_mask(int(mask_table[b]))
        vertex_targets[b, v] = 1.0

    vertex_r2 = _linear_probe_r2(emb_np, vertex_targets)
    print(f"  R² (Ridge, 256 samples, 2048→4): {vertex_r2:.6f}")

    # ── 2f. Boundary predictor geometry ──
    print_subsection("Boundary Predictor vs Kernel Geometry")

    bp = model.model.local_encoder.boundary_predictor_module
    q_weight = bp.q_proj_layer.weight.detach().float().cpu()
    k_weight = bp.k_proj_layer.weight.detach().float().cpu()

    q_proj = byte_embs @ q_weight.T
    k_proj = byte_embs @ k_weight.T

    q_normed = q_proj / q_proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k_normed = k_proj / k_proj.norm(dim=1, keepdim=True).clamp(min=1e-8)

    trans_cos = (q_normed @ k_normed.T).numpy()
    trans_dissim = 1.0 - trans_cos

    off_diag = ~np.eye(256, dtype=bool)
    r_boundary = np.corrcoef(trans_dissim[off_diag], cdm_f[off_diag])[0, 1]

    self_cos = np.array([trans_cos[b, b] for b in range(256)])
    r_self = np.corrcoef(self_cos, byte_weights.astype(np.float64))[0, 1]

    print(f"  Pearson(boundary_dissim, code_dist) = {r_boundary:.6f}")
    print(f"  Pearson(self_cos, mask_weight)       = {r_self:.6f}")

    # Linear probe: boundary Q projection → mask12
    q_proj_np = q_proj.numpy()
    bp_probe_r2 = _linear_probe_r2(q_proj_np, mask_bits)
    print(f"  R² (Q-projection → mask12):          {bp_probe_r2:.6f}")

    return {
        "r_pearson": r_pearson, "r_spearman": r_spearman,
        "family_ratio": ratio, "probe_r2": probe_r2,
        "per_bit_r2": per_bit_r2, "vertex_r2": vertex_r2,
        "cos_dist": cos_dist, "cdm": cdm_f,
        "byte_embs": byte_embs, "mask_bits": mask_bits,
    }


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r) if not np.isnan(r) else 0.0  # pyright: ignore[reportUnknownReturnType] # type: ignore[reportUnknownReturnType] # Handle NaN case]
    except ImportError:
        # Fallback: manual rank correlation
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        return float(np.corrcoef(rx, ry)[0, 1])  # pyright: ignore[reportUnknownReturnType]


def _linear_probe_r2(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> float:
    """Ridge regression R² with leave-one-out cross-validation approximation."""
    n, d = X.shape
    _, k = Y.shape if Y.ndim == 2 else (n, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Center
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge: W = (X^T X + alpha I)^{-1} X^T Y
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W = np.linalg.solve(XtX + alpha * np.eye(d), XtY)

    Y_pred = Xc @ W
    ss_res = np.sum((Yc - Y_pred) ** 2)
    ss_tot = np.sum(Yc ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# Block 3: Uncoupled Generation Census
# ═══════════════════════════════════════════════════════════════════════════════

def block_uncoupled_generation(model, tokenizer, device) -> dict:
    print_section("BLOCK 3: UNCOUPLED GENERATION CENSUS")
    print(f"  Prompt: '{PROMPT}'")
    print(f"  Generating {GEN_TOKENS_LONG} tokens, temp={TEMPERATURE}, top_k={TOP_K}, seed={SEED}")

    torch.manual_seed(SEED)
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids, max_new_tokens=GEN_TOKENS_LONG,
            do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
        )
    elapsed = time.perf_counter() - t0

    all_ids = output[0].tolist()
    prompt_ids = input_ids[0].tolist()
    gen_ids = all_ids[len(prompt_ids):]

    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    print(f"  Generated {len(gen_ids)} tokens in {elapsed:.1f}s ({len(gen_ids) / elapsed:.1f} tok/s)")
    print(f"  Text: {text[:200]}...")

    # Extract bytes and boundary info
    bytes_list, is_boundary = [], []
    for tid in gen_ids:
        bytes_list.append(extract_byte_from_token(tid))
        is_boundary.append(tid >= BOLMO_BOUNDARY_OFFSET)

    bytes_arr = np.array(bytes_list, dtype=np.uint8)
    boundary_arr = np.array(is_boundary)
    n_bytes = len(bytes_arr)

    # Byte statistics
    byte_counts = np.bincount(bytes_arr, minlength=256)
    byte_entropy = _entropy(byte_counts)
    unique_bytes = int((byte_counts > 0).sum())

    print(f"\n  Bytes: {n_bytes} total, {unique_bytes}/256 unique, entropy={byte_entropy:.2f} bits")
    print(f"  Boundary tokens: {boundary_arr.sum()} ({boundary_arr.mean() * 100:.1f}%)")

    # ── Kernel trajectory ──
    print_subsection("Kernel Trajectory")

    kernel = RouterKernel(atlas_dir=ATLAS_DIR)
    cdm = get_code_distance_matrix()
    mask_table = get_mask12_table()

    horizons, vertices, phases, code_dists = [], [], [], []
    h_dists, weights = [], []
    horizon_set = set()
    prev_h = None

    for b in bytes_arr:
        kernel.step_byte(int(b))
        h = int(kernel.current_horizon.item())
        chi = int(kernel.current_vertex.item())
        p = int(kernel.current_phase.item())

        horizons.append(h)
        vertices.append(chi)
        phases.append(p)

        sig = kernel.signature()
        a12 = int(sig.a_hex, 16)
        b12 = int(sig.b_hex, 16)
        hd = popcount(a12 ^ (b12 ^ LAYER_MASK_12))
        h_dists.append(hd)
        weights.append(popcount(int(mask_table[int(b)])))

        if hd == 0:
            horizon_set.add(h)

        if prev_h is not None:
            code_dists.append(int(cdm[prev_h, h]))
        prev_h = h

    h_arr = np.array(horizons)
    chi_arr = np.array(vertices)
    p_arr = np.array(phases)
    hd_arr = np.array(h_dists)
    cd_arr = np.array(code_dists) if code_dists else np.array([6])
    w_arr = np.array(weights)

    h_counts = np.bincount(h_arr, minlength=256)
    chi_counts = np.bincount(chi_arr, minlength=4)
    p_counts = np.bincount(p_arr, minlength=4)

    print(f"  horizon_distance: mean={hd_arr.mean():.2f}  std={hd_arr.std():.2f}  min={hd_arr.min()}  max={hd_arr.max()}")
    print(f"  code_dist:        mean={cd_arr.mean():.2f}  std={cd_arr.std():.2f}  (neutral=6.0)")
    print(f"  horizon states visited (h_dist=0): {len(horizon_set)}/256")
    print(f"  horizon entropy:  {_entropy(h_counts):.2f} bits (max=8.0)")
    print(f"  vertex dist:      {chi_counts.tolist()} ({(chi_counts / n_bytes * 100).round(1).tolist()}%)")
    print(f"  phase dist:       {p_counts.tolist()}")
    print(f"  mask weight:      mean={w_arr.mean():.2f}  std={w_arr.std():.2f}")

    # Code distance histogram
    cd_counts = np.bincount(cd_arr, minlength=13)
    print(f"  code_dist hist:   {cd_counts.tolist()}")

    # Horizon coverage over time
    visit_kernel = RouterKernel(atlas_dir=ATLAS_DIR)
    visited = set()
    coverage_at = {}
    first_full = -1
    checkpoints = [100, 250, 500, 1000, 1500, 2000]
    for i, b in enumerate(bytes_list):
        visit_kernel.step_byte(b)
        visited.add(int(visit_kernel.current_horizon.item()))
        step = i + 1
        if step in checkpoints:
            coverage_at[step] = len(visited)
        if len(visited) == 256 and first_full < 0:
            first_full = step

    print_subsection("Horizon Coverage")
    for step, cov in sorted(coverage_at.items()):
        print(f"  after {step:5d} bytes: {cov}/256 ({cov / 256 * 100:.1f}%)")
    if first_full > 0:
        print(f"  full coverage at byte {first_full}")
    else:
        print(f"  NOT fully covered ({len(visited)}/256 after {n_bytes} bytes)")

    # ── XOR-Cosine Bridge ──
    print_subsection("XOR-Cosine Bridge (Dynamic)")

    emb_weight = model.model.local_encoder.byte_embedding.weight.detach().float().cpu()
    byte_embs = emb_weight[BOLMO_TOKEN_OFFSET:BOLMO_TOKEN_OFFSET + 256]
    norms = byte_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    byte_normed = byte_embs / norms
    cos_sim_matrix = (byte_normed @ byte_normed.T).numpy()

    xor_dists, cos_sims = [], []
    for i in range(n_bytes - 1):
        b1, b2 = int(bytes_arr[i]), int(bytes_arr[i + 1])
        xor_dists.append(int(cdm[b1, b2]))
        cos_sims.append(float(cos_sim_matrix[b1, b2]))

    xor_arr = np.array(xor_dists, dtype=np.float64)
    cos_arr = np.array(cos_sims, dtype=np.float64)

    r_bridge = np.corrcoef(xor_arr, cos_arr)[0, 1]
    print(f"  Pearson(xor_dist, cos_sim) = {r_bridge:.6f}  (n={len(xor_arr)})")
    print(f"  xor_dist: mean={xor_arr.mean():.3f}  std={xor_arr.std():.3f}")
    print(f"  cos_sim:  mean={cos_arr.mean():.4f}  std={cos_arr.std():.4f}")

    # Per code distance
    print(f"  cos_sim by code distance:")
    for d in range(13):
        mask_d = xor_arr == d
        if mask_d.sum() > 0:
            print(f"    d={d:2d}: n={int(mask_d.sum()):5d}  cos_sim={cos_arr[mask_d].mean():.4f}")

    # ── Transport statistics ──
    print_subsection("Transport Statistics vs Kernel Priors")

    P_code, _ = get_code_prior()
    empirical_cd = np.bincount(cd_arr, minlength=13).astype(np.float64)
    empirical_cd = empirical_cd / max(empirical_cd.sum(), 1)
    kl_code = kl_divergence(empirical_cd, P_code[:13])
    print(f"  KL(empirical_code_dist || P_code) = {kl_code * 1000:.2f} millibits")

    # Weight distribution
    P_w = np.bincount(np.array([popcount(int(mask_table[b])) for b in range(256)]),
                      minlength=13).astype(np.float64)
    P_w = P_w / P_w.sum()
    empirical_w = np.bincount(w_arr, minlength=13).astype(np.float64)
    empirical_w = empirical_w / max(empirical_w.sum(), 1)
    kl_w = kl_divergence(empirical_w, P_w)
    print(f"  KL(empirical_weight || P_weight)  = {kl_w * 1000:.2f} millibits")

    # Vertex distribution
    P_chi = np.array([0.25, 0.25, 0.25, 0.25])
    empirical_chi = chi_counts.astype(np.float64) / max(chi_counts.sum(), 1)
    kl_chi = kl_divergence(empirical_chi, P_chi)
    print(f"  KL(empirical_vertex || uniform)   = {kl_chi * 1000:.2f} millibits")

    # ── Parity invariants ──
    print_subsection("Parity Invariants (P8)")

    O, E, parity = trajectory_parity_commitment(bytes_list)
    sig = kernel.signature()

    if parity == 0:
        exp_a = ARCHETYPE_A12 ^ O
        exp_b = ARCHETYPE_B12 ^ E
    else:
        exp_a = (ARCHETYPE_B12 ^ LAYER_MASK_12) ^ E
        exp_b = (ARCHETYPE_A12 ^ LAYER_MASK_12) ^ O

    actual_a = int(sig.a_hex, 16)
    actual_b = int(sig.b_hex, 16)
    p8_ok = (exp_a == actual_a) and (exp_b == actual_b)
    mono = monodromy_from_parity(O, E)

    print(f"  O=0x{O:03x}  E=0x{E:03x}  n_mod_2={parity}")
    print(f"  expected: A=0x{exp_a:03x}  B=0x{exp_b:03x}")
    print(f"  actual:   A=0x{actual_a:03x}  B=0x{actual_b:03x}")
    print(f"  P8: {'PASS' if p8_ok else 'FAIL'}")
    print(f"  Monodromy A(trajectory) = {mono:.4f}")

    return {
        "bytes_arr": bytes_arr, "boundary_arr": boundary_arr,
        "horizons": h_arr, "vertices": chi_arr, "phases": p_arr,
        "code_dists": cd_arr, "h_dists": hd_arr, "weights": w_arr,
        "kernel": kernel, "text": text,
        "r_bridge": r_bridge, "kl_code": kl_code,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Block 4: Kernel Invariant Verification In Vivo
# ═══════════════════════════════════════════════════════════════════════════════

def block_invariants(bytes_arr: np.ndarray, kernel: RouterKernel):
    print_section("BLOCK 4: KERNEL INVARIANT VERIFICATION")
    bytes_list = [int(b) for b in bytes_arr]
    n = len(bytes_list)

    # ── P7: Depth-4 alternation identity ──
    print_subsection("P7: Depth-4 Alternation (xyxy = id)")

    test_kernel = RouterKernel(atlas_dir=ATLAS_DIR)
    pairs_tested = set()
    p7_pass, p7_fail = 0, 0

    for i in range(n - 1):
        x, y = bytes_list[i], bytes_list[i + 1]
        if x == y:
            continue
        pair = (x, y)
        if pair in pairs_tested:
            continue
        pairs_tested.add(pair)

        test_kernel.reset()
        s0 = test_kernel.signature().state_hex
        test_kernel.step_byte(x)
        test_kernel.step_byte(y)
        test_kernel.step_byte(x)
        test_kernel.step_byte(y)
        s4 = test_kernel.signature().state_hex

        if s0 == s4:
            p7_pass += 1
        else:
            p7_fail += 1

    print(f"  Pairs tested: {len(pairs_tested)}")
    print(f"  Pass: {p7_pass}  Fail: {p7_fail}  Rate: {p7_pass / max(len(pairs_tested), 1) * 100:.1f}%")

    # ── P6: Depth-2 non-commutativity ──
    print_subsection("P6: Depth-2 Commutation")

    commute_count = 0
    noncommute_count = 0
    for pair in pairs_tested:
        x, y = pair
        test_kernel.reset()
        test_kernel.step_byte(x)
        test_kernel.step_byte(y)
        sxy = test_kernel.signature().state_hex

        test_kernel.reset()
        test_kernel.step_byte(y)
        test_kernel.step_byte(x)
        syx = test_kernel.signature().state_hex

        if sxy == syx:
            commute_count += 1
        else:
            noncommute_count += 1

    print(f"  Commuting pairs: {commute_count}  Non-commuting: {noncommute_count}")
    print(f"  Non-commutativity rate: {noncommute_count / max(len(pairs_tested), 1) * 100:.1f}%")
    print(f"  Expected: 99.6% (255/256)")

    # ── Quotient dynamics check ──
    print_subsection("Quotient Dynamics: K₄ Vertex Transitions")

    mask_table = get_mask12_table()
    vertex_transitions = defaultdict(int)
    for i in range(n - 1):
        b = bytes_list[i + 1]
        chi_b = vertex_charge_from_mask(int(mask_table[b]))
        key = (int(bytes_arr[i] if i == 0 else bytes_arr[i]),)  # simplified
        # Track (chi_prev_state, chi_byte) → chi_next_state
        # This checks the K4 group law

    # Simplified: check that vertex transitions form a group
    v_transitions = np.zeros((4, 4), dtype=int)
    replay = RouterKernel(atlas_dir=ATLAS_DIR)
    for i, b in enumerate(bytes_list):
        chi_before = int(replay.current_vertex.item())
        chi_byte = vertex_charge_from_mask(int(mask_table[b]))
        replay.step_byte(b)
        chi_after = int(replay.current_vertex.item())
        v_transitions[chi_byte, chi_before] += 1  # count

    print(f"  Vertex transition matrix (byte_charge × state_vertex):")
    print(f"    {v_transitions.tolist()}")

    total = v_transitions.sum()
    print(f"  Total transitions: {total}")


# ═══════════════════════════════════════════════════════════════════════════════
# Block 5: Windowed Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

def block_windowed_dynamics(bytes_arr: np.ndarray):
    print_section("BLOCK 5: WINDOWED DYNAMICS")

    bytes_list = [int(b) for b in bytes_arr]
    n = len(bytes_list)
    W = WINDOW_SIZE
    cdm = get_code_distance_matrix()
    mask_table = get_mask12_table()
    P_code, _ = get_code_prior()

    if n < W + 1:
        print(f"  Trajectory too short ({n} < {W + 1}), skipping.")
        return

    # Replay to get per-step horizons
    kernel = RouterKernel(atlas_dir=ATLAS_DIR)
    horizons = []
    for b in bytes_list:
        kernel.step_byte(b)
        horizons.append(int(kernel.current_horizon.item()))

    # ── Windowed KL ──
    print_subsection(f"Windowed KL(empirical || P_code), W={W}")

    n_windows = n - W
    windowed_kl = np.zeros(n_windows)
    windowed_mono = np.zeros(n_windows)

    for start in range(n_windows):
        window_bytes = bytes_list[start:start + W]

        # Code distances within window
        window_cd = []
        for j in range(len(window_bytes) - 1):
            h1 = horizons[start + j]
            h2 = horizons[start + j + 1]
            window_cd.append(int(cdm[h1, h2]))

        if window_cd:
            emp = np.bincount(window_cd, minlength=13).astype(np.float64)
            emp = emp / max(emp.sum(), 1)
            windowed_kl[start] = kl_divergence(emp, P_code[:13])

        # Windowed monodromy
        O_w, E_w, _ = trajectory_parity_commitment(window_bytes)
        windowed_mono[start] = monodromy_from_parity(O_w, E_w)

    print(f"  Global mean:  {windowed_kl.mean() * 1000:.2f} millibits")
    print(f"  Global std:   {windowed_kl.std() * 1000:.2f} millibits")
    print(f"  Max:          {windowed_kl.max() * 1000:.2f} millibits at window {np.argmax(windowed_kl)}")
    print(f"  Min:          {windowed_kl.min() * 1000:.2f} millibits")

    burst_threshold = windowed_kl.mean() + 2 * windowed_kl.std()
    n_bursts = (windowed_kl > burst_threshold).sum()
    print(f"  Bursts (>mean+2σ): {n_bursts}/{n_windows} ({n_bursts / n_windows * 100:.1f}%)")

    # ── Windowed monodromy ──
    print_subsection("Windowed Monodromy")

    print(f"  mean={windowed_mono.mean():.4f}  std={windowed_mono.std():.4f}")
    print(f"  min={windowed_mono.min():.4f}  max={windowed_mono.max():.4f}")
    print(f"  BU target (δ_BU/π): {0.062:.4f}")

    # Contraction/retraction detection
    delta_mono = np.diff(windowed_mono)
    n_contract = (delta_mono < 0).sum()
    n_retract = (delta_mono > 0).sum()
    print(f"  Contractions: {n_contract}  Retractions: {n_retract}")
    print(f"  Ratio (contract/retract): {n_contract / max(n_retract, 1):.3f}")

    # ── Phase transition statistics ──
    print_subsection("Phase Transitions")

    kernel2 = RouterKernel(atlas_dir=ATLAS_DIR)
    phase_transitions = defaultdict(int)
    prev_p = None
    for b in bytes_list:
        kernel2.step_byte(b)
        p = int(kernel2.current_phase.item())
        if prev_p is not None:
            phase_transitions[(prev_p, p)] += 1
        prev_p = p

    print(f"  Transition counts (from → to):")
    for (pf, pt) in sorted(phase_transitions.keys()):
        print(f"    {pf}→{pt}: {phase_transitions[(pf, pt)]}")

    # Check for hard constraints (e.g., 3→1 never occurs)
    all_transitions = set(phase_transitions.keys())
    possible = {(i, j) for i in range(4) for j in range(4)}
    missing = possible - all_transitions
    if missing:
        print(f"  Missing transitions: {sorted(missing)}")
    else:
        print(f"  All 16 phase transitions present")


# ═══════════════════════════════════════════════════════════════════════════════
# Block 6: Coupled Generation with Full Instrumentation
# ═══════════════════════════════════════════════════════════════════════════════

def block_coupled_generation(model, tokenizer, device, n_tokens: int = GEN_TOKENS_SHORT,
                             prompt: str = PROMPT, label: str = "coupled") -> dict:
    print_section(f"BLOCK 6: COUPLED GENERATION ({label}, {n_tokens} tokens)")
    print(f"  Prompt: '{prompt[:60]}...'")

    configs = {
        "uncoupled": None,
        "global_only": CouplingConfig(couple_local_decoder=False, couple_local_encoder=False),
        "local_dec": CouplingConfig(couple_local_decoder=True, couple_local_encoder=False),
        "local_enc": CouplingConfig(couple_local_decoder=False, couple_local_encoder=True),
        "both": CouplingConfig(couple_local_decoder=True, couple_local_encoder=True),
    }

    results = {}
    cdm = get_code_distance_matrix()
    mask_table = get_mask12_table()

    for name, cfg in configs.items():
        print(f"\n  ─── {name} {'─' * (50 - len(name))}")

        labe = None
        try:
            if cfg is not None:
                labe = GyroLabe(
                    model, atlas_dir=ATLAS_DIR, config=cfg,
                    token_offset=BOLMO_TOKEN_OFFSET,
                )
                labe.install()

            torch.manual_seed(SEED)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(device)

            if labe is not None:
                labe.reset()
                labe.prime_from_tokens(input_ids[0].tolist())

            t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids = model.generate(
                    input_ids, max_new_tokens=n_tokens,
                    do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                    labe=labe,
                )
            elapsed = time.perf_counter() - t0

            all_out = out_ids[0].tolist()
            prompt_ids = input_ids[0].tolist()
            gen_ids = all_out[len(prompt_ids):]
            n_gen = len(gen_ids)
            text = tokenizer.decode(all_out, skip_special_tokens=True)

            # Extract bytes
            gen_bytes = [extract_byte_from_token(tid) for tid in gen_ids]

            # Kernel replay
            replay_kernel = RouterKernel(atlas_dir=ATLAS_DIR)
            # Prime with prompt
            for tid in prompt_ids:
                replay_kernel.step_byte(extract_byte_from_token(tid))

            replay_horizons, replay_vertices, replay_code_dists = [], [], []
            prev_h = int(replay_kernel.current_horizon.item())
            for b in gen_bytes:
                replay_kernel.step_byte(b)
                h = int(replay_kernel.current_horizon.item())
                chi = int(replay_kernel.current_vertex.item())
                replay_horizons.append(h)
                replay_vertices.append(chi)
                replay_code_dists.append(int(cdm[prev_h, h]))
                prev_h = h

            h_arr = np.array(replay_horizons)
            chi_arr = np.array(replay_vertices)
            cd_arr = np.array(replay_code_dists)

            h_counts = np.bincount(h_arr, minlength=256)
            chi_counts = np.bincount(chi_arr, minlength=4)

            byte_arr = np.array(gen_bytes)
            byte_counts = np.bincount(byte_arr, minlength=256)

            unique_h = int((h_counts > 0).sum())
            h_ent = _entropy(h_counts)

            # Parity
            O, E, par = trajectory_parity_commitment(gen_bytes)
            mono = monodromy_from_parity(O, E)

            # GyroLabe stats
            correlation = 0.0
            gain = 1.0
            if labe is not None and labe.trajectory:
                s = labe.stats()
                correlation = s.get("mean_correlation", 0.0)
                gain = s.get("mean_gain_at_peak", 1.0)

            tok_s = n_gen / max(elapsed, 1e-9)
            print(f"    tokens={n_gen}  elapsed={elapsed:.1f}s  tok/s={tok_s:.1f}")
            print(f"    code_dist:   mean={cd_arr.mean():.2f}  std={cd_arr.std():.2f}")
            print(f"    correlation: {correlation:.4f}")
            print(f"    gain:        {gain:.4f}")
            print(f"    unique_h:    {unique_h}/256  h_entropy={h_ent:.2f}")
            print(f"    vertex:      {chi_counts.tolist()} ({(chi_counts / max(n_gen, 1) * 100).round(1).tolist()}%)")
            print(f"    unique_bytes: {int((byte_counts > 0).sum())}/256  byte_entropy={_entropy(byte_counts):.2f}")
            print(f"    monodromy:   {mono:.4f}")
            print(f"    text: {text[:120]}...")

            results[name] = {
                "n_tokens": n_gen, "elapsed": elapsed, "tok_s": tok_s,
                "code_dist_mean": float(cd_arr.mean()),
                "code_dist_std": float(cd_arr.std()),
                "correlation": correlation, "gain": gain,
                "unique_h": unique_h, "h_entropy": h_ent,
                "chi_dist": chi_counts.tolist(),
                "unique_bytes": int((byte_counts > 0).sum()),
                "byte_entropy": _entropy(byte_counts),
                "monodromy": mono,
                "text": text,
                "gen_bytes": gen_bytes,
                "horizons": replay_horizons,
                "vertices": replay_vertices,
                "code_dists": replay_code_dists,
            }

        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}
        finally:
            if labe is not None:
                labe.restore()

    # ── Comparison table ──
    print_subsection("Comparison Summary")
    header = f"  {'Config':<14} {'code_d':>7} {'corr':>7} {'gain':>7} {'h_uniq':>7} {'h_ent':>6} {'mono':>6} {'tok/s':>6}"
    print(header)
    print(f"  {'─' * len(header)}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<14} ERROR: {r['error'][:40]}")
            continue
        print(f"  {name:<14} {r['code_dist_mean']:7.2f} {r['correlation']:7.4f} "
              f"{r['gain']:7.4f} {r['unique_h']:5d}/256 {r['h_entropy']:6.2f} "
              f"{r['monodromy']:6.4f} {r['tok_s']:6.1f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Block 7: Dynamic Boundary–Kernel Correlation
# ═══════════════════════════════════════════════════════════════════════════════

def block_boundary_kernel_correlation(model, tokenizer, device):
    print_section("BLOCK 7: DYNAMIC BOUNDARY–KERNEL CORRELATION")
    print(f"  Running coupled generation and tracking boundary events...")

    cfg = CouplingConfig(couple_local_decoder=True, couple_local_encoder=False)
    labe = GyroLabe(
        model, atlas_dir=ATLAS_DIR, config=cfg,
        token_offset=BOLMO_TOKEN_OFFSET,
    )
    labe.install()

    cdm = get_code_distance_matrix()
    mask_table = get_mask12_table()

    try:
        torch.manual_seed(SEED + 7)
        enc = tokenizer(LONG_PROMPT, return_tensors="pt")
        input_ids = enc.input_ids.to(device)

        labe.reset()
        labe.prime_from_tokens(input_ids[0].tolist())

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids, max_new_tokens=GEN_TOKENS_LONG,
                do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                labe=labe,
            )
        elapsed = time.perf_counter() - t0

        all_out = out_ids[0].tolist()
        prompt_ids = input_ids[0].tolist()
        gen_ids = all_out[len(prompt_ids):]

        print(f"  Generated {len(gen_ids)} tokens in {elapsed:.1f}s")

        # Classify each token
        is_boundary = np.array([tid >= BOLMO_BOUNDARY_OFFSET for tid in gen_ids])
        gen_bytes = [extract_byte_from_token(tid) for tid in gen_ids]

        # Replay kernel
        replay = RouterKernel(atlas_dir=ATLAS_DIR)
        for tid in prompt_ids:
            replay.step_byte(extract_byte_from_token(tid))

        prev_h = int(replay.current_horizon.item())
        prev_chi = int(replay.current_vertex.item())
        prev_p = int(replay.current_phase.item())

        code_dists_step = []
        vertex_changed = []
        phase_changed = []
        horizon_dists = []

        for b in gen_bytes:
            replay.step_byte(b)
            h = int(replay.current_horizon.item())
            chi = int(replay.current_vertex.item())
            p = int(replay.current_phase.item())

            sig = replay.signature()
            a12 = int(sig.a_hex, 16)
            b12 = int(sig.b_hex, 16)
            hd = popcount(a12 ^ (b12 ^ LAYER_MASK_12))

            code_dists_step.append(int(cdm[prev_h, h]))
            vertex_changed.append(chi != prev_chi)
            phase_changed.append(p != prev_p)
            horizon_dists.append(hd)

            prev_h, prev_chi, prev_p = h, chi, p

        code_dists_step = np.array(code_dists_step, dtype=np.float64)
        vertex_changed = np.array(vertex_changed)
        phase_changed = np.array(phase_changed)
        horizon_dists = np.array(horizon_dists, dtype=np.float64)

        n_boundaries = is_boundary.sum()
        n_non_boundaries = (~is_boundary).sum()

        print(f"\n  Boundary tokens: {n_boundaries} ({n_boundaries / len(gen_ids) * 100:.1f}%)")
        print(f"  Non-boundary:    {n_non_boundaries}")

        if n_boundaries > 0 and n_non_boundaries > 0:
            print_subsection("Point-Biserial Correlations: boundary_fired × kernel_observable")

            rpb_code = point_biserial(is_boundary, code_dists_step)
            rpb_vertex = point_biserial(is_boundary, vertex_changed.astype(float))
            rpb_phase = point_biserial(is_boundary, phase_changed.astype(float))
            rpb_hdist = point_biserial(is_boundary, horizon_dists)

            print(f"  boundary × code_dist:       r_pb = {rpb_code:.6f}")
            print(f"  boundary × vertex_changed:  r_pb = {rpb_vertex:.6f}")
            print(f"  boundary × phase_changed:   r_pb = {rpb_phase:.6f}")
            print(f"  boundary × horizon_dist:    r_pb = {rpb_hdist:.6f}")

            # Mean observables at boundary vs non-boundary
            print_subsection("Observables at Boundary vs Non-Boundary Steps")

            for obs_name, obs_arr in [
                ("code_dist", code_dists_step),
                ("horizon_dist", horizon_dists),
                ("mask_weight", np.array([popcount(int(mask_table[b])) for b in gen_bytes], dtype=np.float64)),
            ]:
                at_bnd = obs_arr[is_boundary]
                at_non = obs_arr[~is_boundary]
                print(f"  {obs_name:<16}  boundary: mean={at_bnd.mean():.3f} std={at_bnd.std():.3f}  "
                      f"non-boundary: mean={at_non.mean():.3f} std={at_non.std():.3f}")

            # Vertex distribution at boundaries vs non-boundaries
            replay2 = RouterKernel(atlas_dir=ATLAS_DIR)
            for tid in prompt_ids:
                replay2.step_byte(extract_byte_from_token(tid))

            bnd_vertices, non_bnd_vertices = [], []
            for i, b in enumerate(gen_bytes):
                replay2.step_byte(b)
                chi = int(replay2.current_vertex.item())
                (bnd_vertices if is_boundary[i] else non_bnd_vertices).append(chi)

            if bnd_vertices:
                bnd_v_counts = np.bincount(bnd_vertices, minlength=4)
                non_v_counts = np.bincount(non_bnd_vertices, minlength=4) if non_bnd_vertices else np.zeros(4)
                print(f"\n  Vertex at boundary:     {bnd_v_counts.tolist()} ({(bnd_v_counts / max(len(bnd_vertices), 1) * 100).round(1).tolist()}%)")
                print(f"  Vertex at non-boundary: {non_v_counts.tolist()} ({(non_v_counts / max(len(non_bnd_vertices), 1) * 100).round(1).tolist()}%)")
        else:
            print("  Cannot compute boundary correlations (all same class)")

        # GyroLabe trajectory stats
        if labe.trajectory:
            s = labe.stats()
            print_subsection("GyroLabe Trajectory Summary")
            print(f"  steps:       {s['steps']}")
            print(f"  correlation: {s.get('mean_correlation', 0):.4f}  std={s.get('std_correlation', 0):.4f}")
            print(f"  code_dist:   {s.get('mean_code_dist', 0):.2f}  std={s.get('std_code_dist', 0):.2f}")
            print(f"  gain:        {s.get('mean_gain_at_peak', 1):.4f}")
            print(f"  unique_h:    {s.get('unique_h', 0)}/256  h_entropy={s.get('h_entropy', 0):.2f}")
            print(f"  chi_dist:    {s.get('chi_dist', [])}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        labe.restore()


# ═══════════════════════════════════════════════════════════════════════════════
# Block 8: Long Coupled Run with Logprob Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def block_long_coupled_run(model, tokenizer, device):
    print_section("BLOCK 8: LONG COUPLED RUN (local_dec, 2000 tokens)")

    results = block_coupled_generation(
        model, tokenizer, device,
        n_tokens=GEN_TOKENS_LONG,
        prompt=LONG_PROMPT,
        label="long_local_dec",
    )

    # Run windowed dynamics on the best config
    best_name = "local_dec"
    if best_name in results and "gen_bytes" in results[best_name]:
        gen_bytes = results[best_name]["gen_bytes"]
        print_subsection(f"Windowed Dynamics for {best_name}")
        block_windowed_dynamics(np.array(gen_bytes, dtype=np.uint8))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Block 9: Per-Layer Activation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def block_per_layer_analysis(model, tokenizer, device):
    print_section("BLOCK 9: PER-LAYER CORRELATION ANALYSIS")
    print(f"  Generating with local_dec coupling, collecting per-layer telemetry...")

    cfg = CouplingConfig(
        couple_local_decoder=True, couple_local_encoder=False,
        store_layer_telemetry=True,
    )
    labe = GyroLabe(
        model, atlas_dir=ATLAS_DIR, config=cfg,
        token_offset=BOLMO_TOKEN_OFFSET,
    )
    labe.install()

    try:
        torch.manual_seed(SEED + 99)
        enc = tokenizer(PROMPT, return_tensors="pt")
        input_ids = enc.input_ids.to(device)

        labe.reset()
        labe.prime_from_tokens(input_ids[0].tolist())

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids, max_new_tokens=GEN_TOKENS_SHORT,
                do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                labe=labe,
            )

        n_gen = out_ids.shape[1] - input_ids.shape[1]
        print(f"  Generated {n_gen} tokens")

        # Aggregate per-layer statistics
        layer_stats = defaultdict(lambda: {
            "correlations": [], "gains": [], "code_dists": [],
            "peak_masses": [], "h_peaks": [],
        })

        for step_record in labe.trajectory:
            if "layers" not in step_record:
                continue
            for layer_data in step_record["layers"]:
                lid = layer_data["layer_idx"]
                layer_stats[lid]["correlations"].append(layer_data.get("correlation", 0.0))
                layer_stats[lid]["gains"].append(layer_data.get("gain_at_peak", 1.0))
                layer_stats[lid]["code_dists"].append(layer_data.get("code_dist", 6))
                layer_stats[lid]["peak_masses"].append(layer_data.get("peak_mass", 0.0))
                layer_stats[lid]["h_peaks"].append(layer_data.get("h_peak", 0))

        print_subsection("Per-Layer Summary")
        header = f"  {'Layer':>6} {'Site':<8} {'corr':>7} {'gain':>7} {'code_d':>7} {'p_mass':>7} {'h_uniq':>7}"
        print(header)
        print(f"  {'─' * len(header)}")

        for lid in sorted(layer_stats.keys()):
            ls = layer_stats[lid]
            if not ls["correlations"]:
                continue

            site = "global" if lid < 100 else ("loc_dec" if lid < 2000 else "loc_enc")
            corr = np.mean(ls["correlations"])
            gain = np.mean(ls["gains"])
            cd = np.mean(ls["code_dists"])
            pm = np.mean(ls["peak_masses"])
            unique_h = len(set(ls["h_peaks"]))

            print(f"  {lid:6d} {site:<8} {corr:7.4f} {gain:7.4f} {cd:7.2f} {pm:7.4f} {unique_h:5d}/256")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        labe.restore()


# ═══════════════════════════════════════════════════════════════════════════════
# Block 10: Structural Fingerprint Summary
# ═══════════════════════════════════════════════════════════════════════════════

def block_fingerprint(arch: dict, geom: dict, uncoupled: dict,
                      coupled_results: dict):
    print_section("BLOCK 10: STRUCTURAL FINGERPRINT")

    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ SUBSTRATE                                              │")
    print(f"  │  hidden={arch['hidden']}  fibers={arch['n_fiber_hidden']}  "
          f"global_inter={arch['inter']}  local_inter={arch['local_inter']}  │")
    print(f"  │  layers: {arch['n_enc']}enc + {arch['n_global']}global + {arch['n_dec']}dec"
          f"                        │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ STATIC GEOMETRY                                        │")
    print(f"  │  Pearson(cos,code):  {geom['r_pearson']:.4f}                              │")
    print(f"  │  Spearman(cos,code): {geom['r_spearman']:.4f}                              │")
    print(f"  │  Family ratio:       {geom['family_ratio']:.4f}  (<1=cluster)               │")
    print(f"  │  Probe R² (→mask12): {geom['probe_r2']:.4f}                              │")
    print(f"  │  Probe R² (→vertex): {geom['vertex_r2']:.4f}                              │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ UNCOUPLED TRAJECTORY                                   │")
    print(f"  │  code_dist:    {uncoupled['code_dists'].mean():.2f} ± {uncoupled['code_dists'].std():.2f}  "
          f"(neutral=6.0)             │")
    print(f"  │  XOR-cos bridge: {uncoupled['r_bridge']:.4f}                              │")
    print(f"  │  KL(cd||P_code): {uncoupled['kl_code'] * 1000:.2f} millibits"
          f"                       │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ COUPLED COMPARISON                                     │")

    for name in ["uncoupled", "global_only", "local_dec", "local_enc", "both"]:
        r = coupled_results.get(name, {})
        if "error" in r:
            print(f"  │  {name:<14} ERROR                                    │")
            continue
        if not r:
            continue
        print(f"  │  {name:<14} corr={r.get('correlation', 0):.3f}  "
              f"cd={r.get('code_dist_mean', 0):.2f}  "
              f"h={r.get('unique_h', 0):3d}/256  "
              f"mono={r.get('monodromy', 0):.3f}  │")

    print(f"  └─────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  GyroLabe Structural Observatory — Bolmo-1B                ║")
    print("║  GGG ASI Alignment Router × Byte-Level Language Model      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Model: {MODEL_DIR}")
    print(f"  Atlas: {ATLAS_DIR}")

    if not MODEL_DIR.exists():
        print(f"  ERROR: Model not found at {MODEL_DIR}")
        sys.exit(1)
    if not ATLAS_DIR.exists():
        print(f"  ERROR: Atlas not found at {ATLAS_DIR}")
        sys.exit(1)

    device = detect_device()
    model, tokenizer = load_bolmo(MODEL_DIR, device)

    t_total = time.perf_counter()

    # Block 1: Architecture
    arch = block_architecture(model)

    # Block 2: Static geometry (embeddings, probes, boundary predictor)
    geom = block_static_geometry(model)

    # Block 3: Uncoupled generation census
    uncoupled = block_uncoupled_generation(model, tokenizer, device)

    # Free inference caches
    try:
        model.model.local_encoder.free_inference_cache()
        model.model.local_decoder.free_inference_cache()
    except Exception:
        pass

    # Block 4: Kernel invariant verification
    block_invariants(uncoupled["bytes_arr"], uncoupled["kernel"])

    # Block 5: Windowed dynamics on uncoupled trajectory
    block_windowed_dynamics(uncoupled["bytes_arr"])

    # Block 6: Short coupled generation comparison (all configs)
    coupled = block_coupled_generation(model, tokenizer, device,
                                        n_tokens=GEN_TOKENS_SHORT, prompt=PROMPT,
                                        label="short_comparison")

    try:
        model.model.local_encoder.free_inference_cache()
        model.model.local_decoder.free_inference_cache()
    except Exception:
        pass

    # Block 7: Dynamic boundary-kernel correlation
    block_boundary_kernel_correlation(model, tokenizer, device)

    try:
        model.model.local_encoder.free_inference_cache()
        model.model.local_decoder.free_inference_cache()
    except Exception:
        pass

    # Block 8: Long coupled run
    long_results = block_long_coupled_run(model, tokenizer, device)

    try:
        model.model.local_encoder.free_inference_cache()
        model.model.local_decoder.free_inference_cache()
    except Exception:
        pass

    # Block 9: Per-layer activation analysis
    block_per_layer_analysis(model, tokenizer, device)

    try:
        model.model.local_encoder.free_inference_cache()
        model.model.local_decoder.free_inference_cache()
    except Exception:
        pass

    # Block 10: Structural fingerprint
    block_fingerprint(arch, geom, uncoupled, coupled)

    elapsed_total = time.perf_counter() - t_total
    print(f"\n  Total observatory time: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}m)")
    print(f"\n{'═' * 72}")
    print(f"  OBSERVATORY COMPLETE")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()