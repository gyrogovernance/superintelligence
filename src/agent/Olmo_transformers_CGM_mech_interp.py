"""
OLMo transformer MI and CGM physics probes.


Structural probes (established invariants, run with --mi-cgm-struct):
- CGM-1: MLP 256x43 structure (holographic horizon)
- CGM-2: Layer-4 closure pattern ([s,s,s,F] x 8)
- CGM-E: Byte feature correlation (adapter viability)
- CGM-H3: Horizon leakage by layer type

Adapter probes (active research, run with --mi-cgm):
- CGM-A: Attention horizon enrichment (normalized)
- CGM-C: Best horizon layer with enrichment
- CGM-D3: K4 parity discovery
- CGM-J: Next-token horizon alignment (LM vs decoder)

Archived (see Mech_Interp_Report.md):
- CGM-3, B, F, G3: findings documented, wrong targets

Usage:
  python -m src.agent.Olmo_transformers_CGM_mech_interp             # generation test
  python -m src.agent.Olmo_transformers_CGM_mech_interp --mi        # entropy + top-k
  python -m src.agent.Olmo_transformers_CGM_mech_interp --mi-cgm    # adapter probes
  python -m src.agent.Olmo_transformers_CGM_mech_interp --mi-struct # structural probes
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.agent.adapters import SemanticTokenCodec


def _configure_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


def format_chat_prompt(user_message: str) -> str:
    """Format user message using OLMo's chat template."""
    return (
        "<|im_start|>system\n"
        "You are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ==========
# CGM Probes (Core)
# ==========

def probe_mlp_intermediate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    layer_idx: int = 16,
) -> dict[str, Any]:
    """
    CGM-1: Probe 256x43 structure in MLP intermediate.
    
    11008 = 256 x 43 = holographic horizon x fiber dimension
    """
    layer = model.model.layers[layer_idx]  # type: ignore[attr-defined]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)  # type: ignore[operator]
    
    h = outputs.hidden_states[layer_idx][0].float()
    h_normed = layer.post_attention_layernorm(
        h.to(layer.post_attention_layernorm.weight.dtype)
    )
    
    with torch.no_grad():
        gate = layer.mlp.gate_proj(h_normed)
        up = layer.mlp.up_proj(h_normed)
        intermediate = torch.nn.functional.silu(gate) * up
    
    acts = intermediate.float().cpu()
    seq_len, dim = acts.shape
    
    if dim != 11008:
        return {"layer": layer_idx, "error": f"dim={dim}, expected 11008"}
    
    acts_256x43 = acts.reshape(seq_len, 256, 43)
    channel_norms = acts_256x43.norm(dim=2)
    
    channel_probs = torch.softmax(channel_norms.mean(dim=0), dim=0)
    entropy = -(channel_probs * torch.log(channel_probs + 1e-10)).sum().item()
    max_entropy = float(np.log(256))
    
    max_norm = channel_norms.max(dim=1, keepdim=True).values
    active_mask = channel_norms > (0.1 * max_norm)
    avg_active = active_mask.float().sum(dim=1).mean().item()
    
    return {
        "layer": layer_idx,
        "channel_entropy": float(entropy),
        "max_entropy": max_entropy,
        "entropy_ratio": entropy / max_entropy,
        "avg_active_channels": avg_active,
    }


def probe_layer4_closure(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """
    CGM-2: Test [sliding, sliding, sliding, full] x 8 pattern.
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)  # type: ignore[operator]
    
    states = [h[0].float() for h in outputs.hidden_states]
    
    updates: list[float] = []
    for i in range(1, len(states)):
        delta = states[i] - states[i - 1]
        updates.append(delta.norm().item())
    
    full_layers = [3, 7, 11, 15, 19, 23, 27, 31]
    sliding_layers = [i for i in range(32) if i not in full_layers]
    
    full_updates = [updates[i] for i in full_layers if i < len(updates)]
    sliding_updates = [updates[i] for i in sliding_layers if i < len(updates)]
    
    return {
        "full_mean": float(np.mean(full_updates)) if full_updates else 0.0,
        "full_std": float(np.std(full_updates)) if full_updates else 0.0,
        "sliding_mean": float(np.mean(sliding_updates)) if sliding_updates else 0.0,
        "sliding_std": float(np.std(sliding_updates)) if sliding_updates else 0.0,
        "ratio": float(np.mean(full_updates)) / (float(np.mean(sliding_updates)) + 1e-8),
    }


# ==========
# Integration Probes
# ==========

def probe_attention_horizon_alignment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
    layer_idx: int = 16,
) -> dict[str, Any]:
    """
    CGM-A: Does attention respect Gyro's 256 semantic horizon buckets?
    
    Now computes enrichment over random baseline to avoid conflating
    semantics with availability.
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)  # type: ignore[attr-defined]
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        try:
            outputs = model(input_tensor, output_attentions=True)  # type: ignore[operator]
        except Exception as e:
            return {"error": str(e)}
    
    if outputs.attentions is None:
        return {"error": "attentions=None"}
    
    attn = outputs.attentions[layer_idx][0].float()
    seq = attn.shape[1]
    
    attn_mean = attn.mean(dim=0)
    last = seq - 1
    weights = attn_mean[last, :last]
    weights = weights / (weights.sum() + 1e-8)
    
    b0s: list[int] = []
    for tid in input_ids[:last]:
        if 0 <= tid < codec.vocab_size:
            b0s.append(codec.encode(tid)[0])
        else:
            b0s.append(0)
    b0s_arr = np.array(b0s, dtype=np.uint8)
    
    last_tid = input_ids[last]
    last_b0 = codec.encode(last_tid)[0] if 0 <= last_tid < codec.vocab_size else 0
    
    same_mask = b0s_arr == last_b0
    n_same = int(same_mask.sum())
    n_diff = int((~same_mask).sum())
    same_mass = float(weights[same_mask].sum().item()) if same_mask.any() else 0.0
    
    # Baseline: what random attention would give
    baseline = n_same / last if last > 0 else 0.0
    enrichment = same_mass / baseline if baseline > 0 else 0.0
    
    return {
        "layer": layer_idx,
        "same_horizon_mass": round(same_mass, 3),
        "baseline": round(baseline, 3),
        "enrichment": round(enrichment, 3),
        "n_same": n_same,
        "n_diff": n_diff,
    }


def probe_best_horizon_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
) -> dict[str, Any]:
    """
    CGM-C: Find which layer's attention best respects semantic horizon.
    
    Now reports enrichment over random baseline per layer.
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)  # type: ignore[attr-defined]
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        try:
            outputs = model(input_tensor, output_attentions=True)  # type: ignore[operator]
        except Exception as e:
            return {"error": str(e)}
    
    if outputs.attentions is None:
        return {"error": "attentions=None"}
    
    last_tid = input_ids[-1]
    last_b0 = codec.encode(last_tid)[0] if 0 <= last_tid < codec.vocab_size else 0
    
    b0s = np.array([
        codec.encode(tid)[0] if 0 <= tid < codec.vocab_size else 0
        for tid in input_ids[:-1]
    ], dtype=np.uint8)
    same_mask = b0s == last_b0
    n_same = int(same_mask.sum())
    last = len(input_ids) - 1
    baseline = n_same / last if last > 0 else 0.0
    
    # (layer, same_mass, enrichment)
    results: list[tuple[int, float, float]] = []
    for layer_idx, attn_layer in enumerate(outputs.attentions):
        attn = attn_layer[0].float().mean(dim=0)
        weights = attn[-1, :-1]
        weights = weights / (weights.sum() + 1e-8)
        same_mass = float(weights[same_mask].sum().item()) if same_mask.any() else 0.0
        enrich = same_mass / baseline if baseline > 0 else 0.0
        results.append((layer_idx, same_mass, enrich))
    
    best_layer, best_mass, best_enrich = max(results, key=lambda x: x[2])
    
    return {
        "per_layer": results,
        "best_layer": best_layer,
        "best_mass": round(best_mass, 3),
        "best_enrichment": round(best_enrich, 3),
        "baseline": round(baseline, 3),
        "n_same": n_same,
        "n_total": last,
    }


def probe_byte_feature_correlation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_message: str,
    atlas_dir: Path = Path("data/atlas"),
) -> dict[str, Any]:
    """CGM-E: Correlate hidden states with Gyro's 43-dim byte features."""
    phen_path = atlas_dir / "phenomenology.npz"
    if not phen_path.exists():
        return {"error": f"phenomenology.npz not found"}
    
    with np.load(phen_path) as phen:
        if "features_K43" not in phen:
            return {"error": "features_K43 not in phenomenology.npz"}
        byte_features = phen["features_K43"]
    
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)  # type: ignore[attr-defined]
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)  # type: ignore[operator]
    
    final = outputs.hidden_states[-1][0].float()
    X = final - final.mean(dim=0, keepdim=True)
    
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # type: ignore[attr-defined]
    except Exception as e:
        return {"error": str(e)}
    
    n_comp = min(43, Vh.shape[0])
    proj = (X @ Vh[:n_comp].T).cpu().numpy()
    bf_proj = byte_features[:, :n_comp] if n_comp <= 43 else byte_features
    
    correlations: list[tuple[int, float]] = []
    bf_norm = bf_proj / (np.linalg.norm(bf_proj, axis=1, keepdims=True) + 1e-8)
    
    for pos in range(proj.shape[0]):
        pos_vec = proj[pos]
        pos_norm = np.linalg.norm(pos_vec)
        if pos_norm < 1e-8:
            correlations.append((0, 0.0))
            continue
        pos_vec = pos_vec / pos_norm
        cos_sims = bf_norm @ pos_vec
        correlations.append((int(np.argmax(cos_sims)), float(cos_sims.max())))
    
    return {
        "avg_byte_correlation": float(np.mean([c[1] for c in correlations])),
        "position_correlations": [round(c[1], 3) for c in correlations[:10]],
    }


# ==========
# Helpers
# ==========

def _code32_bits(codec: SemanticTokenCodec, tid: int) -> int:
    """Pack 4 semantic bytes into 32-bit int."""
    b0, b1, b2, b3 = codec.encode(tid)
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


def _parity32(x: int) -> int:
    """Parity of 32-bit int."""
    return bin(x).count('1') & 1


def _charge_from_masks(code32: int, m0: int, m1: int) -> int:
    """Compute 2-bit K4 charge from two parity masks."""
    b0 = _parity32(code32 & m0)
    b1 = _parity32(code32 & m1)
    return (b1 << 1) | b0


def probe_k4_parity_discovery(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
    layer_idx: int = 4,
) -> dict[str, Any]:
    """
    CGM-D2: Discover K4 parity checks in semantic-code space.
    
    Searches ~528 candidate masks to find pair that maximizes
    charge conservation under attention transport.
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)  # type: ignore[attr-defined]
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        try:
            outputs = model(input_tensor, output_attentions=True)  # type: ignore[operator]
        except Exception as e:
            return {"error": str(e)}
    
    if outputs.attentions is None:
        return {"error": "attentions=None"}
    
    attn = outputs.attentions[layer_idx][0].float().mean(dim=0)
    last = attn.shape[0] - 1
    w = attn[last, :last]
    w = w / (w.sum() + 1e-8)
    
    codes = [_code32_bits(codec, tid) for tid in input_ids[:last]]
    codes_last = _code32_bits(codec, input_ids[last])
    
    # Build candidate masks: single-bit + two-bit XOR
    masks: list[int] = []
    for i in range(32):
        masks.append(1 << i)
    for i in range(32):
        for j in range(i + 1, 32):
            masks.append((1 << i) ^ (1 << j))
    
    # Stage 1: rank masks by entropy
    scored: list[tuple[float, int]] = []
    for m in masks:
        p0 = sum(float(w[k].item()) for k, c in enumerate(codes) if _parity32(c & m) == 0)
        p1 = 1.0 - p0
        ent = 0.0
        if p0 > 1e-10:
            ent -= p0 * np.log(p0)
        if p1 > 1e-10:
            ent -= p1 * np.log(p1)
        scored.append((ent, m))
    scored.sort(reverse=True)
    top_masks = [m for _, m in scored[:48]]
    
    # Stage 2: find best pair
    best: tuple[float, int, int, float, list[float], float] | None = None
    for m0 in top_masks:
        for m1 in top_masks:
            if m1 == m0:
                continue
            last_chi = _charge_from_masks(codes_last, m0, m1)
            
            mismatch = 0.0
            counts = [0.0, 0.0, 0.0, 0.0]
            for k, c in enumerate(codes):
                chi = _charge_from_masks(c, m0, m1)
                counts[chi] += float(w[k].item())
                if chi != last_chi:
                    mismatch += float(w[k].item())
            
            mass_entropy = -sum(p * np.log(p + 1e-10) for p in counts)
            score = mismatch + 0.05 * (np.log(4) - mass_entropy)
            
            if best is None or score < best[0]:
                best = (score, m0, m1, mismatch, counts, mass_entropy)
    
    if best is None:
        return {"error": "no valid mask pair found"}
    
    _, m0, m1, mismatch, counts, mass_entropy = best
    
    return {
        "layer": layer_idx,
        "mask0": hex(m0),
        "mask1": hex(m1),
        "attn_mismatch": float(mismatch),
        "charge_mass": [round(x, 3) for x in counts],
        "charge_entropy": float(mass_entropy),
    }


def _load_or_build_b0_prototypes(
    model: AutoModelForCausalLM,
    codec: SemanticTokenCodec,
    path: Path,
) -> np.ndarray:
    """Build 256 prototypes: mean embedding per b0 bucket."""
    if path.exists():
        return np.load(path)
    
    embed = model.model.embed_tokens.weight.detach().float().cpu().numpy()  # type: ignore[attr-defined]
    protos = np.zeros((256, embed.shape[1]), dtype=np.float32)
    counts = np.zeros(256, dtype=np.int64)
    
    for tid in range(codec.vocab_size):
        b0 = codec.encode(tid)[0]
        protos[b0] += embed[tid]
        counts[b0] += 1
    
    for b0 in range(256):
        if counts[b0] > 0:
            protos[b0] /= counts[b0]
    
    np.save(path, protos)
    return protos


def probe_b0_decode_from_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
    layer_idx: int = 4,
) -> dict[str, Any]:
    """
    CGM-G2: Decode semantic b0 from hidden state via nearest prototype.
    
    This is the correct semantic horizon decoder.
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)  # type: ignore[attr-defined]
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)  # type: ignore[operator]
    
    h = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
    
    proto_path = Path("data/models/Olmo-3-7B-Instruct/b0_prototypes.npy")
    protos = _load_or_build_b0_prototypes(model, codec, proto_path)
    
    # Cosine similarity
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    p_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    
    scores = h_norm @ p_norm.T
    pred_b0 = np.argmax(scores, axis=1)
    
    true_b0 = np.array([codec.encode(tid)[0] for tid in input_ids], dtype=np.int64)
    
    acc = float((pred_b0 == true_b0).mean())
    top5 = np.argsort(scores, axis=1)[:, -5:]
    top5_acc = float(np.mean([true_b0[i] in top5[i] for i in range(len(true_b0))]))
    
    return {
        "layer": layer_idx,
        "acc": acc,
        "top5_acc": top5_acc,
        "pred_b0": pred_b0[:10].tolist(),
        "true_b0": true_b0[:10].tolist(),
    }


def probe_horizon_leakage_by_layer_type(
    per_layer_results: list[tuple[int, float]],
) -> dict[str, Any]:
    """
    CGM-H2: Quantify horizon alignment leakage by layer type.
    
    Uses CGM-C per_layer results.
    """
    full_layers = {3, 7, 11, 15, 19, 23, 27, 31}
    
    full = [m for l, m in per_layer_results if l in full_layers]
    sliding = [m for l, m in per_layer_results if l not in full_layers]
    
    full_mean = float(np.mean(full)) if full else 0.0
    sliding_mean = float(np.mean(sliding)) if sliding else 0.0
    
    return {
        "full_mean": full_mean,
        "sliding_mean": sliding_mean,
        "full_minus_sliding": full_mean - sliding_mean,
    }


# ==========
# V3 Probes (Refined)
# ==========

def probe_k4_balanced_discovery(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
    layer_idx: int = 4,
    max_mismatch: float = 0.1,
) -> dict[str, Any]:
    """
    CGM-D3: Discover balanced K4 (high entropy, low mismatch).
    
    Stage 1: filter candidates with mismatch < max_mismatch
    Stage 2: among those, maximize charge entropy
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        try:
            outputs = model(input_tensor, output_attentions=True)
        except Exception as e:
            return {"error": str(e)}
    
    if outputs.attentions is None:
        return {"error": "attentions=None"}
    
    attn = outputs.attentions[layer_idx][0].float().mean(dim=0)
    last = attn.shape[0] - 1
    w = attn[last, :last]
    w = w / (w.sum() + 1e-8)
    
    codes = [_code32_bits(codec, tid) for tid in input_ids[:last]]
    codes_last = _code32_bits(codec, input_ids[last])
    
    # Same candidate masks as D2
    masks: list[int] = []
    for i in range(32):
        masks.append(1 << i)
    for i in range(32):
        for j in range(i + 1, 32):
            masks.append((1 << i) ^ (1 << j))
    
    # Stage 1: filter by mismatch
    candidates: list[tuple[int, int, float, list[float]]] = []
    for m0 in masks:
        for m1 in masks:
            if m1 == m0:
                continue
            last_chi = _charge_from_masks(codes_last, m0, m1)
            mismatch = 0.0
            counts = [0.0, 0.0, 0.0, 0.0]
            for k, c in enumerate(codes):
                chi = _charge_from_masks(c, m0, m1)
                counts[chi] += float(w[k].item())
                if chi != last_chi:
                    mismatch += float(w[k].item())
            if mismatch < max_mismatch:
                candidates.append((m0, m1, mismatch, counts))
    
    if not candidates:
        return {"error": f"no candidates with mismatch < {max_mismatch}"}
    
    # Stage 2: maximize entropy
    def entropy_of(counts: list[float]) -> float:
        return -sum(p * np.log(p + 1e-10) for p in counts)
    
    best = max(candidates, key=lambda x: entropy_of(x[3]))
    m0, m1, mismatch, counts = best
    
    mass_entropy = entropy_of(counts)
    
    return {
        "layer": layer_idx,
        "n_candidates": len(candidates),
        "mask0": hex(m0),
        "mask1": hex(m1),
        "mismatch": float(mismatch),
        "charge_mass": [round(x, 3) for x in counts],
        "charge_entropy": round(float(mass_entropy), 3),
        "max_entropy": round(float(np.log(4)), 3),
    }


def probe_weighted_b0_decoder(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
    layer_idx: int = 4,
    k4_masks: tuple[int, int] = (0x10200000, 0x10001000),
) -> dict[str, Any]:
    """
    CGM-G3: Weight prototypes by inverse frequency to improve b0 decoding.
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)
    
    h = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
    
    embed = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    proto_path = Path("data/models/Olmo-3-7B-Instruct/b0_prototypes_weighted.npy")
    
    if proto_path.exists():
        protos = np.load(proto_path)
    else:
        protos = np.zeros((256, embed.shape[1]), dtype=np.float32)
        counts = np.zeros(256, dtype=np.float64)
        
        for tid in range(codec.vocab_size):
            b0 = codec.encode(tid)[0]
            weight = 1.0 / (counts[b0] + 1.0)
            protos[b0] += weight * embed[tid]
            counts[b0] += weight
        
        for b0 in range(256):
            if counts[b0] > 0:
                protos[b0] /= counts[b0]
        
        np.save(proto_path, protos)
        protos = np.load(proto_path)
    
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    p_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    
    scores = h_norm @ p_norm.T
    pred_b0 = np.argmax(scores, axis=1)
    
    true_b0 = np.array([codec.encode(tid)[0] for tid in input_ids], dtype=np.int64)
    
    acc = float((pred_b0 == true_b0).mean())
    top5 = np.argsort(scores, axis=1)[:, -5:]
    top5_acc = float(np.mean([true_b0[i] in top5[i] for i in range(len(true_b0))]))
    
    baseline_top5 = 0.083
    
    return {
        "layer": layer_idx,
        "acc": round(acc, 3),
        "top5_acc": round(top5_acc, 3),
        "improvement": round(top5_acc - baseline_top5, 3),
        "pred_b0": pred_b0[:10].tolist(),
        "true_b0": true_b0[:10].tolist(),
    }


def probe_horizon_leakage_refined(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
) -> dict[str, Any]:
    """
    CGM-H3: Refined horizon leakage by layer type.
    
    Leakage = 1 - same_horizon_mass (decay from L4 peak).
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        try:
            outputs = model(input_tensor, output_attentions=True)
        except Exception as e:
            return {"error": str(e)}
    
    if outputs.attentions is None:
        return {"error": "attentions=None"}
    
    last_tid = input_ids[-1]
    last_b0 = codec.encode(last_tid)[0] if 0 <= last_tid < codec.vocab_size else 0
    
    b0s = np.array([
        codec.encode(tid)[0] if 0 <= tid < codec.vocab_size else 0
        for tid in input_ids[:-1]
    ], dtype=np.uint8)
    same_mask = b0s == last_b0
    
    full_layers = [3, 7, 11, 15, 19, 23, 27, 31]
    
    full_leakage: list[float] = []
    sliding_leakage: list[float] = []
    
    for layer_idx, attn_layer in enumerate(outputs.attentions):
        attn = attn_layer[0].float().mean(dim=0)
        weights = attn[-1, :-1]
        weights = weights / (weights.sum() + 1e-8)
        same_mass = float(weights[same_mask].sum().item()) if same_mask.any() else 0.0
        leakage = 1.0 - same_mass
        
        if layer_idx in full_layers:
            full_leakage.append(leakage)
        else:
            sliding_leakage.append(leakage)
    
    full_mean = float(np.mean(full_leakage)) if full_leakage else 0.0
    sliding_mean = float(np.mean(sliding_leakage)) if sliding_leakage else 0.0
    
    return {
        "full_leakage_mean": round(full_mean, 3),
        "sliding_leakage_mean": round(sliding_mean, 3),
        "full_minus_sliding": round(full_mean - sliding_mean, 4),
    }


# ==========
# CGM-J: Next-Token Horizon Alignment
# ==========

def probe_next_token_horizon(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    codec: SemanticTokenCodec,
    user_message: str,
    layer_idx: int = 4,
) -> dict[str, Any]:
    """
    CGM-J: Next-token horizon alignment.
    
    Compares:
    1. LM logits -> probability mass per b0 bucket (what transformer predicts)
    2. Hidden state prototypes -> similarity per b0 bucket (what adapter sees)
    
    This is the key test: is next-token horizon encoded in hidden state?
    """
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)
    
    # 1. LM logits -> b0 distribution
    logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    
    # Aggregate by b0 bucket
    lm_b0_mass = np.zeros(256, dtype=np.float64)
    for tid in range(min(len(probs), codec.vocab_size)):
        b0 = codec.encode(tid)[0]
        lm_b0_mass[b0] += probs[tid]
    
    # 2. Hidden state -> b0 similarity via prototypes
    h = outputs.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
    
    proto_path = Path("data/models/Olmo-3-7B-Instruct/b0_prototypes.npy")
    if not proto_path.exists():
        return {"error": "b0_prototypes.npy not found - run G2 first"}
    protos = np.load(proto_path)
    
    h_norm = h / (np.linalg.norm(h) + 1e-8)
    p_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    similarities = p_norm @ h_norm
    decoder_b0_mass = np.exp(similarities * 5)  # temperature scaling
    decoder_b0_mass /= decoder_b0_mass.sum()
    
    # 3. Compare distributions
    # Top-k overlap
    lm_top10 = set(np.argsort(lm_b0_mass)[-10:])
    dec_top10 = set(np.argsort(decoder_b0_mass)[-10:])
    top10_overlap = len(lm_top10 & dec_top10)
    
    # KL divergence (LM || decoder)
    eps = 1e-10
    kl_div = float(np.sum(lm_b0_mass * np.log((lm_b0_mass + eps) / (decoder_b0_mass + eps))))
    
    # Correlation
    corr = float(np.corrcoef(lm_b0_mass, decoder_b0_mass)[0, 1])
    
    # Top LM bucket
    lm_top_b0 = int(np.argmax(lm_b0_mass))
    lm_top_mass = float(lm_b0_mass[lm_top_b0])
    dec_rank_of_lm_top = int(256 - np.searchsorted(
        np.sort(decoder_b0_mass), decoder_b0_mass[lm_top_b0]
    ))
    
    return {
        "layer": layer_idx,
        "top10_overlap": top10_overlap,
        "kl_divergence": round(kl_div, 3),
        "correlation": round(corr, 3),
        "lm_top_b0": lm_top_b0,
        "lm_top_mass": round(lm_top_mass, 3),
        "decoder_rank_of_lm_top": dec_rank_of_lm_top,
    }


# ==========
# Analysis Runners
# ==========

def _load_model_and_codec() -> tuple[Any, Any, SemanticTokenCodec, Path]:
    """Common model/tokenizer/codec loading."""
    MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
    CODEC_PATH = MODEL_DIR / "gyro_codebook.npz"
    
    print("\nLoading model (eager attention)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    
    embed_tokens = model.model.embed_tokens.weight.detach().float()
    if CODEC_PATH.exists():
        codec = SemanticTokenCodec.load(CODEC_PATH)
    else:
        codec = SemanticTokenCodec.build(embed_tokens, model.config.vocab_size)
        codec.save(CODEC_PATH)
    
    return model, tokenizer, codec, MODEL_DIR


def run_cgm_struct_analysis() -> None:
    """Structural probes: CGM-1, CGM-2, CGM-E, CGM-H3 (established invariants)."""
    _configure_stdout_utf8()
    
    print("CGM Structural Probes")
    print("=" * 10)
    
    model, tokenizer, codec, _ = _load_model_and_codec()
    
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Intermediate: {model.config.intermediate_size} = 256 x 43")
    
    user_message = "Freedom is not worth having if it does not include the freedom to make mistakes."
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    
    print(f"\nPrompt: {user_message[:50]}...")
    print(f"Tokens: {len(input_ids)}")
    
    # CGM-1: MLP 256x43
    print("\n" + "-" * 10)
    print("[CGM-1] MLP Intermediate (256 x 43)")
    print("-" * 10)
    
    for layer in [8, 16, 24]:
        stats = probe_mlp_intermediate(model, input_tensor, layer)
        if "error" not in stats:
            print(f"  L{layer}: entropy={stats['channel_entropy']:.2f}/{stats['max_entropy']:.2f} "
                  f"({stats['entropy_ratio']:.0%}), active={stats['avg_active_channels']:.0f}/256")
    
    # CGM-2: Layer-4 closure
    print("\n" + "-" * 10)
    print("[CGM-2] Layer-4 Closure Pattern")
    print("-" * 10)
    
    closure = probe_layer4_closure(model, input_tensor)
    print(f"  Full attention:    {closure['full_mean']:.2f} +/- {closure['full_std']:.2f}")
    print(f"  Sliding attention: {closure['sliding_mean']:.2f} +/- {closure['sliding_std']:.2f}")
    print(f"  Ratio:             {closure['ratio']:.2f}")
    
    # CGM-E: Byte feature correlation
    print("\n" + "-" * 10)
    print("[CGM-E] Byte Feature Correlation")
    print("-" * 10)
    
    bf = probe_byte_feature_correlation(model, tokenizer, user_message)
    if "error" not in bf:
        print(f"  Avg correlation:  {bf['avg_byte_correlation']:.3f}")
    
    # CGM-H3: Horizon leakage
    print("\n" + "-" * 10)
    print("[CGM-H3] Horizon Leakage by Layer Type")
    print("-" * 10)
    
    h3 = probe_horizon_leakage_refined(model, tokenizer, codec, user_message)
    if "error" not in h3:
        print(f"  Full leakage:    {h3['full_leakage_mean']:.3f}")
        print(f"  Sliding leakage: {h3['sliding_leakage_mean']:.3f}")
        print(f"  Full - Sliding:  {h3['full_minus_sliding']:.4f}")
    
    print("\n" + "=" * 10)
    print("Structural probes complete. See Mech_Interp_Report.md for analysis.")


def run_cgm_mi_analysis() -> None:
    """Adapter probes: CGM-A, CGM-C, CGM-D3, CGM-J (active research)."""
    _configure_stdout_utf8()
    
    print("CGM Adapter Probes")
    print("=" * 10)
    
    model, tokenizer, codec, _ = _load_model_and_codec()
    
    print(f"  Layers: {model.config.num_hidden_layers}")
    
    user_message = "Freedom is not worth having if it does not include the freedom to make mistakes."
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    
    print(f"\nPrompt: {user_message[:50]}...")
    print(f"Tokens: {len(input_ids)}")
    
    # CGM-A: Attention horizon enrichment
    print("\n" + "-" * 10)
    print("[CGM-A] Attention Horizon Enrichment")
    print("-" * 10)
    
    att = probe_attention_horizon_alignment(model, tokenizer, codec, user_message)
    if "error" in att:
        print(f"  ERROR: {att['error']}")
    else:
        print(f"  Same-horizon mass: {att['same_horizon_mass']:.3f}")
        print(f"  Random baseline:   {att['baseline']:.3f} ({att['n_same']}/{att['n_same']+att['n_diff']})")
        print(f"  Enrichment:        {att['enrichment']:.2f}x")
    
    # CGM-C: Best horizon layer with enrichment
    print("\n" + "-" * 10)
    print("[CGM-C] Best Horizon Layer (Enrichment)")
    print("-" * 10)
    
    best = probe_best_horizon_layer(model, tokenizer, codec, user_message)
    if "error" in best:
        print(f"  ERROR: {best['error']}")
    else:
        print(f"  Baseline: {best['baseline']:.3f} ({best['n_same']}/{best['n_total']} same)")
        print(f"  Best layer: L{best['best_layer']} (enrich={best['best_enrichment']:.2f}x)")
        prog = [(l, f"{e:.2f}x") for l, _, e in best['per_layer'][::4]]
        print(f"  Progression: {prog}")
    
    # CGM-D3: K4 parity
    print("\n" + "-" * 10)
    print("[CGM-D3] K4 Parity Discovery")
    print("-" * 10)
    
    d3 = probe_k4_balanced_discovery(model, tokenizer, codec, user_message)
    if "error" in d3:
        print(f"  ERROR: {d3['error']}")
    else:
        print(f"  Candidates (mm<0.1): {d3['n_candidates']}")
        print(f"  Best masks: {d3['mask0']}, {d3['mask1']}")
        print(f"  Mismatch:   {d3['mismatch']:.3f}")
        print(f"  Charge mass: {d3['charge_mass']}")
        ent_pct = 100 * d3['charge_entropy'] / d3['max_entropy']
        print(f"  Entropy:    {d3['charge_entropy']:.2f} ({ent_pct:.0f}% of max)")
    
    # CGM-J: Next-token horizon alignment
    print("\n" + "-" * 10)
    print("[CGM-J] Next-Token Horizon Alignment")
    print("-" * 10)
    
    j = probe_next_token_horizon(model, tokenizer, codec, user_message)
    if "error" in j:
        print(f"  ERROR: {j['error']}")
    else:
        print(f"  LM top b0: {j['lm_top_b0']} (mass={j['lm_top_mass']:.3f})")
        print(f"  Decoder rank of LM top: {j['decoder_rank_of_lm_top']}/256")
        print(f"  Top-10 overlap: {j['top10_overlap']}/10")
        print(f"  Correlation:    {j['correlation']:.3f}")
        print(f"  KL divergence:  {j['kl_divergence']:.3f}")
    
    # Summary
    print("\n" + "=" * 10)
    print("SUMMARY")
    print("=" * 10)
    if "enrichment" in att:
        print(f"  Horizon enrichment: {att['enrichment']:.2f}x over baseline")
    if "best_enrichment" in best:
        print(f"  Best read layer: L{best['best_layer']} ({best['best_enrichment']:.2f}x)")
    if "charge_entropy" in d3:
        ent_pct = 100 * d3['charge_entropy'] / d3['max_entropy']
        print(f"  K4 entropy: {ent_pct:.0f}% (skewed)")
    if "correlation" in j:
        print(f"  Next-token horizon corr: {j['correlation']:.3f}")


def run_mi_analysis() -> None:
    """Standard MI: entropy, top-k tokens."""
    _configure_stdout_utf8()
    
    MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
    CODEC_PATH = MODEL_DIR / "gyro_codebook.npz"
    
    print("MI Analysis")
    print("=" * 10)
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    embed_tokens = model.model.embed_tokens.weight.detach().float()  # type: ignore[attr-defined]
    if CODEC_PATH.exists():
        codec = SemanticTokenCodec.load(CODEC_PATH)
    else:
        codec = SemanticTokenCodec.build(embed_tokens, model.config.vocab_size)
        codec.save(CODEC_PATH)
    
    user_message = "Freedom is not worth having if it does not include the freedom to make mistakes."
    print(f"\nUSER: {user_message}")
    
    formatted = format_chat_prompt(user_message)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        outputs = model(input_tensor, return_dict=True)  # type: ignore[operator]
    
    logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    print(f"\n[MI-1] Entropy: {entropy:.2f} bits")
    
    print("\n[MI-2] Top-10 Tokens:")
    top_probs, top_ids = torch.topk(probs, 10)
    for i, (p, tid) in enumerate(zip(top_probs, top_ids)):
        tid_int = int(tid.item())
        token_str = tokenizer.decode([tid_int])
        code = codec.encode(tid_int)
        print(f"  {i+1}. {p.item():.3f} | [{code[0]:02x},{code[1]:02x}] | {repr(token_str)}")


def run_test() -> None:
    """Generation sanity check."""
    _configure_stdout_utf8()
    
    MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
    
    print("OLMo Generation Test")
    print("=" * 10)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    user_message = "Freedom is not worth having if it does not include the freedom to make mistakes."
    print(f"\nUSER: {user_message}")
    
    formatted = format_chat_prompt(user_message)
    inputs = tokenizer(formatted, return_tensors="pt")
    
    import time
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nASSISTANT: {response}")
    print(f"\n({time.time()-start:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        flag = sys.argv[1]
        if flag == "--mi":
            run_mi_analysis()
        elif flag == "--mi-cgm":
            run_cgm_mi_analysis()
        elif flag == "--mi-struct":
            run_cgm_struct_analysis()
        else:
            print(f"Unknown: {flag}")
            print("Usage: --mi | --mi-cgm (adapter) | --mi-struct (structural)")
    else:
        run_test()
