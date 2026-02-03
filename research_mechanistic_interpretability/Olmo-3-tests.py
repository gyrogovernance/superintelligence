# research_mechanistic_interpretability/Olmo-3-tests-v3.py
"""
Fixed CGM Kernel-Grounded MI with proper diversity.

Key fixes:
1. Use chat template to get structural token diversity
2. Use LONG prompts (500+ tokens) to ensure horizon collisions
3. Query from positions with MULTIPLE horizon matches
4. Proper Q/K capture hooks
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Callable
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.adapters import SemanticTokenCodec
from src.router.kernel import RouterKernel
from src.router.constants import (
    ARCHETYPE_A12, ARCHETYPE_B12, LAYER_MASK_12, C_PERP_12,
    unpack_state,
)


# =============================================================================
# Kernel Trace with Collision Analysis
# =============================================================================

@dataclass
class KernelTraceWithCollisions:
    """Extended trace with collision information."""
    horizons: np.ndarray        # (n,) horizon at each position
    vertices: np.ndarray        # (n,) K4 vertex
    phases: np.ndarray          # (n,) phase
    u: np.ndarray               # (n,) u-code
    v: np.ndarray               # (n,) v-code
    
    # Collision info
    horizon_counts: dict[int, int]          # horizon -> count
    positions_by_horizon: dict[int, list[int]]  # horizon -> list of positions
    
    # Diversity metrics
    n_unique_horizons: int
    max_collision_size: int
    entropy: float  # entropy of horizon distribution


def kernel_trace_with_collisions(
    codec: SemanticTokenCodec,
    input_ids: list[int],
    atlas_dir: Path,
) -> KernelTraceWithCollisions:
    """Compute kernel trace with collision analysis."""
    K = RouterKernel(atlas_dir)
    n = len(input_ids)
    
    horizons = np.zeros(n, dtype=np.int64)
    vertices = np.zeros(n, dtype=np.int64)
    phases = np.zeros(n, dtype=np.int64)
    u_codes = np.zeros(n, dtype=np.int64)
    v_codes = np.zeros(n, dtype=np.int64)
    
    for t, tid in enumerate(input_ids):
        bs = codec.encode(int(tid))
        for b in bs:
            K.step_byte(b)
        
        s24 = int(K.ontology[K.state_index])
        a12, b12 = unpack_state(s24)
        
        horizons[t] = K.current_horizon
        vertices[t] = K.current_vertex
        phases[t] = K.current_phase
        u_codes[t] = (a12 ^ ARCHETYPE_A12) & LAYER_MASK_12
        v_codes[t] = (b12 ^ ARCHETYPE_B12) & LAYER_MASK_12
    
    # Build collision structures
    horizon_counts: dict[int, int] = {}
    positions_by_horizon: dict[int, list[int]] = {}
    
    for t, h in enumerate(horizons):
        h_int = int(h)
        horizon_counts[h_int] = horizon_counts.get(h_int, 0) + 1
        if h_int not in positions_by_horizon:
            positions_by_horizon[h_int] = []
        positions_by_horizon[h_int].append(t)
    
    n_unique = len(horizon_counts)
    max_collision = max(horizon_counts.values()) if horizon_counts else 0
    
    # Compute entropy
    probs = np.array(list(horizon_counts.values())) / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    return KernelTraceWithCollisions(
        horizons=horizons,
        vertices=vertices,
        phases=phases,
        u=u_codes,
        v=v_codes,
        horizon_counts=horizon_counts,
        positions_by_horizon=positions_by_horizon,
        n_unique_horizons=n_unique,
        max_collision_size=max_collision,
        entropy=entropy,
    )


# =============================================================================
# Q/K/V Capture (Fixed)
# =============================================================================

class QKVCapture:
    """Capture Q, K, V by hooking the projection outputs."""
    
    def __init__(self):
        self.Q: Optional[torch.Tensor] = None
        self.K: Optional[torch.Tensor] = None
        self.V: Optional[torch.Tensor] = None
        self._handles: list[Any] = []
    
    def attach(self, layer_module):
        """Attach hooks to q_proj, k_proj, v_proj."""
        def make_hook(name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
            def hook(module, inputs, outputs):
                if name == "q":
                    self.Q = outputs.detach().float().cpu()
                elif name == "k":
                    self.K = outputs.detach().float().cpu()
                elif name == "v":
                    self.V = outputs.detach().float().cpu()
            return hook
        
        attn = layer_module.self_attn
        self._handles.append(attn.q_proj.register_forward_hook(make_hook("q")))
        self._handles.append(attn.k_proj.register_forward_hook(make_hook("k")))
        self._handles.append(attn.v_proj.register_forward_hook(make_hook("v")))
    
    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# =============================================================================
# Find Good Query Positions
# =============================================================================

def find_query_positions_with_collisions(
    trace: KernelTraceWithCollisions,
    min_matches: int = 3,
) -> list[tuple[int, int, int]]:
    """
    Find positions where querying would have multiple horizon matches.
    
    Returns: [(query_pos, horizon, n_matches), ...]
    """
    results = []
    
    for qpos in range(1, len(trace.horizons)):
        h = int(trace.horizons[qpos])
        # Count how many prior positions share this horizon
        prior_matches = [p for p in trace.positions_by_horizon.get(h, []) if p < qpos]
        n_matches = len(prior_matches)
        
        if n_matches >= min_matches:
            results.append((qpos, h, n_matches))
    
    # Sort by number of matches (descending)
    results.sort(key=lambda x: -x[2])
    return results


# =============================================================================
# Attention Analysis at Collision Points
# =============================================================================

def analyze_attention_at_collision(
    attentions: list[torch.Tensor],  # [layer] -> (heads, seq, seq)
    trace: KernelTraceWithCollisions,
    query_pos: int,
    layer: int,
    head: int,
) -> dict[str, Any]:
    """
    Analyze how attention distributes over horizon-matched positions.
    """
    hq = int(trace.horizons[query_pos])
    prior_positions = trace.positions_by_horizon.get(hq, [])
    matching_positions = np.array([p for p in prior_positions if p < query_pos])
    
    if len(matching_positions) == 0:
        return {"error": "no matching positions"}
    
    attn_row = attentions[layer][head, query_pos, :query_pos].numpy()
    attn_row = attn_row / (attn_row.sum() + 1e-12)
    
    # Mass on matching vs non-matching
    all_positions = np.arange(query_pos)
    non_matching = np.setdiff1d(all_positions, matching_positions)
    
    mass_matching = float(attn_row[matching_positions].sum())
    mass_non_matching = float(attn_row[non_matching].sum()) if len(non_matching) > 0 else 0.0
    
    # Expected baseline
    baseline_matching = len(matching_positions) / query_pos
    enrichment = mass_matching / baseline_matching if baseline_matching > 0 else 0.0
    
    # Distribution within matching positions
    within_matching = attn_row[matching_positions]
    within_matching_norm = within_matching / (within_matching.sum() + 1e-12)
    
    # Uniformity within matching (entropy)
    entropy_within = float(-np.sum(within_matching_norm * np.log2(within_matching_norm + 1e-10)))
    max_entropy = np.log2(len(matching_positions)) if len(matching_positions) > 1 else 0.0
    uniformity = entropy_within / max_entropy if max_entropy > 0 else 1.0
    
    # Position bias within matching
    distances = query_pos - matching_positions
    corr_with_distance = float(np.corrcoef(within_matching, distances)[0, 1]) if len(matching_positions) > 2 else 0.0
    
    return {
        "n_matching": len(matching_positions),
        "n_non_matching": len(non_matching),
        "mass_matching": mass_matching,
        "mass_non_matching": mass_non_matching,
        "baseline": baseline_matching,
        "enrichment": enrichment,
        "uniformity_within": uniformity,
        "recency_correlation": corr_with_distance,  # Negative = prefers recent
        "matching_positions": matching_positions.tolist(),
        "matching_weights": within_matching.tolist(),
    }


# =============================================================================
# Walsh Spectrum on Attention-Weighted Distribution
# =============================================================================

def walsh_spectrum_of_attention(
    attentions: list[torch.Tensor],
    trace: KernelTraceWithCollisions,
    query_pos: int,
    layer: int,
    head: int,
) -> dict[str, Any]:
    """
    Compute Walsh spectrum of attention weighted by u-code.
    
    This tests whether attention's spectral energy concentrates on C⊥.
    """
    attn_row = attentions[layer][head, query_pos, :query_pos].numpy()
    attn_row = attn_row / (attn_row.sum() + 1e-12)
    
    # Aggregate attention by u-code
    f = np.zeros(1 << 12, dtype=np.float64)
    for t in range(query_pos):
        u = int(trace.u[t]) & 0xFFF
        f[u] += attn_row[t]
    
    # Walsh-Hadamard transform
    F = f.copy()
    h = 1
    while h < len(F):
        for i in range(0, len(F), h * 2):
            x = F[i:i+h].copy()
            y = F[i+h:i+2*h].copy()
            F[i:i+h] = x + y
            F[i+h:i+2*h] = x - y
        h *= 2
    
    # Energy analysis
    energy_total = float(np.sum(F * F))
    energy_cperp = sum(F[int(s)]**2 for s in C_PERP_12)
    concentration = energy_cperp / (energy_total + 1e-30)
    
    # Top modes
    energies = F * F
    top_indices = np.argsort(energies)[-10:][::-1]
    top_modes = [(int(i), hex(i), float(energies[i])) for i in top_indices]
    
    # Check if top modes are in C⊥
    top_in_cperp = sum(1 for i, _, _ in top_modes if i in C_PERP_12)
    
    return {
        "energy_total": energy_total,
        "energy_cperp": energy_cperp,
        "concentration": concentration,
        "expected_random": len(C_PERP_12) / 4096,
        "concentration_ratio": concentration / (len(C_PERP_12) / 4096),
        "top_modes": top_modes,
        "top_in_cperp": top_in_cperp,
    }


# =============================================================================
# Long Prompt Generator
# =============================================================================

LONG_PROMPTS = {
    "philosophy": """
The Common Governance Model demonstrates that coherent recursive measurement requires specific 
structural properties. Starting from five constraints expressed in modal logic, the framework 
derives three-dimensional space with six degrees of freedom as the unique solution satisfying 
operational coherence requirements.

Authority, understood as the legitimate capacity to determine operational outcomes, requires 
constitutional principles invariant across contexts. In physical measurement, observers maintain 
descriptive authority while subject to the same laws. In artificial intelligence, decision 
processes must preserve legitimate authority while operating autonomously. Both domains present 
the same fundamental question: what structural requirements determine coherent authority?

Constitutional principles function as invariant constraints determining all subsequent structure, 
distinguishing foundational necessities from contingent choices. This document presents the 
Common Governance Model (CGM), which establishes structural requirements from a single 
foundational axiom: operational structure must trace to a shared source.

The framework treats governance as mathematical structure by specifying the minimal conditions 
required for operations to preserve shared authority while maintaining necessary distinctions. 
The model is "common" because the same logical requirements apply wherever coherent authority 
must be maintained. In physical systems this manifests as conservation laws traceable to a 
unified origin. In informational systems it requires that all processing states remain 
reachable from a designated reference.

The foundational axiom, termed "The Source is Common" (CS), establishes that right transitions 
preserve the reference state while left transitions alter it, creating fundamental chirality. 
From this chiral seed, four additional constraints specify the axiom's requirements at 
increasing modal depths.

Non-absolute unity (UNA) prevents homogeneous collapse at depth two. Non-absolute opposition 
(ONA) prevents absolute contradiction at the same depth. Balanced closure (BU-Egress) achieves 
commutative closure at depth four. Memory reconstruction (BU-Ingress) ensures the balanced 
state reconstructs all prior conditions.
""",
    
    "technical": """
The kernel operates on a 24-bit state split into two 12-bit components, A and B. The archetype 
state has A equal to 0xAAA and B equal to 0x555. These components are bitwise complements.

The transition law proceeds in defined steps. Given a current state and an input byte, the 
kernel first computes the intron by XORing the byte with 0xAA. It then expands the intron to 
a 12-bit mask for the A component using a canonical expansion function. The B mask is always 
zero. The kernel mutates A by XORing it with the mask. Finally, the kernel performs gyration: 
the next A becomes B XOR 0xFFF, and the next B becomes the mutated A XOR 0xFFF.

This transition law is bijective on the ontology for each byte. Every state has exactly one 
predecessor and one successor under each byte operation. The reference byte 0xAA produces a 
zero mask and acts as an involution: applying it twice returns to the original state. The 
inverse of any byte operation is computable as a conjugation by the reference byte.

The kernel exhibits a depth-four identity: for any two bytes x and y, applying the sequence 
x, y, x, y returns to the original state. This closure property is verified for all ordered 
byte pairs. The trajectory of any byte sequence can be computed in closed form from the XOR 
parity of masks at odd and even positions.

The horizon set contains exactly 256 states satisfying the condition A equals B XOR 0xFFF. 
The one-step neighbourhood of the horizon under all 256 byte actions covers the entire 
ontology. This is verified exhaustively.

The holographic dictionary provides the encoding. For any state with components A and B, the 
corresponding horizon state has components A and A XOR 0xFFF. The byte is the unique preimage 
of the mask A XOR (B XOR 0xFFF) under the expansion function.
""",
}


def format_chat_prompt(user_message: str) -> str:
    """Format with chat template."""
    return (
        "<|im_start|>system\n"
        "You are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# =============================================================================
# Main Runner
# =============================================================================

def run_fixed_mi(
    model_dir: Path,
    atlas_dir: Path,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time
    
    t0 = time.time()
    print("CGM Kernel-Grounded MI v3 (Fixed)")
    print("=" * 50)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    
    # Load codec
    codec_path = model_dir / "gyro_codebook.npz"
    codec = SemanticTokenCodec.load(codec_path)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    
    for name, user_msg in LONG_PROMPTS.items():
        print("=" * 50)
        print(f"[{name}] {len(user_msg)} chars")
        print("=" * 50)
        
        # Format with template
        full_prompt = format_chat_prompt(user_msg)
        input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        n = len(input_ids)
        print(f"Tokens: {n}")
        
        # Compute kernel trace with collisions
        trace = kernel_trace_with_collisions(codec, input_ids, atlas_dir)
        print(f"\n[DIVERSITY] Unique horizons: {trace.n_unique_horizons}/256")
        print(f"[DIVERSITY] Max collision: {trace.max_collision_size} tokens share same horizon")
        print(f"[DIVERSITY] Entropy: {trace.entropy:.2f} bits (max={np.log2(256):.2f})")
        print(f"[DIVERSITY] Vertices: {np.bincount(trace.vertices.astype(np.int32), minlength=4).tolist()}")
        
        # Find positions with collisions
        collision_queries = find_query_positions_with_collisions(trace, min_matches=3)
        print(f"\n[COLLISIONS] {len(collision_queries)} positions have 3+ prior horizon matches")
        
        if len(collision_queries) == 0:
            print("[SKIP] No collision points found, skipping attention analysis")
            continue
        
        # Take best collision point
        best_qpos, best_h, best_n = collision_queries[0]
        print(f"[COLLISIONS] Best: pos={best_qpos}, horizon={best_h}, matches={best_n}")
        
        # Forward pass
        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            out = model(input_tensor, output_attentions=True)
        
        raw_attentions = out.attentions
        if raw_attentions is None:
            print("[SKIP] No attentions returned")
            continue
        attentions = [a[0].float().cpu() for a in raw_attentions]
        
        # Find best head at this collision point
        print(f"\n[ATTENTION] Analyzing at query_pos={best_qpos}")
        
        best_layer, best_head, best_enrich = -1, -1, 0.0
        for layer in range(len(attentions)):
            for head in range(attentions[layer].shape[0]):
                res = analyze_attention_at_collision(
                    attentions, trace, best_qpos, layer, head
                )
                if "error" not in res and res["enrichment"] > best_enrich:
                    best_enrich = res["enrichment"]
                    best_layer = layer
                    best_head = head
        
        print(f"[ATTENTION] Best head: L{best_layer} H{best_head} enrich={best_enrich:.2f}x")
        
        # Detailed analysis at best head
        if best_layer >= 0:
            res = analyze_attention_at_collision(
                attentions, trace, best_qpos, best_layer, best_head
            )
            print(f"[ATTENTION] Mass on {res['n_matching']} matches: {res['mass_matching']:.3f}")
            print(f"[ATTENTION] Mass on {res['n_non_matching']} non-matches: {res['mass_non_matching']:.3f}")
            print(f"[ATTENTION] Baseline: {res['baseline']:.4f}, Enrichment: {res['enrichment']:.2f}x")
            print(f"[ATTENTION] Uniformity within matches: {res['uniformity_within']:.3f}")
            print(f"[ATTENTION] Recency correlation: {res['recency_correlation']:.3f}")
            
            # Show matching positions
            print(f"[MATCHING] Positions: {res['matching_positions'][:10]}...")
            weights = res['matching_weights'][:10]
            print(f"[MATCHING] Weights: {[round(w, 3) for w in weights]}...")
            
            # Walsh spectrum
            walsh = walsh_spectrum_of_attention(
                attentions, trace, best_qpos, best_layer, best_head
            )
            print(f"\n[WALSH] C⊥ concentration: {walsh['concentration']:.4f}")
            print(f"[WALSH] Expected random: {walsh['expected_random']:.4f}")
            print(f"[WALSH] Concentration ratio: {walsh['concentration_ratio']:.2f}x")
            print(f"[WALSH] Top 3 modes: {walsh['top_modes'][:3]}")
            print(f"[WALSH] Top modes in C⊥: {walsh['top_in_cperp']}/10")
        
        # Q/K capture at best layer
        if best_layer >= 0:
            print(f"\n[Q/K] Capturing at L{best_layer}...")
            qkv = QKVCapture()
            qkv.attach(model.model.layers[best_layer])
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            if qkv.Q is not None:
                Q = qkv.Q[0].numpy()  # (seq, hidden)
                
                # Test horizon decodability from Q
                from sklearn.neighbors import KNeighborsClassifier
                y = trace.horizons
                
                # Simple k-NN test
                clf = KNeighborsClassifier(n_neighbors=1)
                # Leave-one-out
                correct = 0
                for i in range(len(y)):
                    train_idx = [j for j in range(len(y)) if j != i]
                    clf.fit(Q[train_idx], y[train_idx])
                    pred = clf.predict(Q[i:i+1])[0]
                    if pred == y[i]:
                        correct += 1
                
                acc = correct / len(y)
                chance = 1.0 / trace.n_unique_horizons
                print(f"[Q/K] Q-based horizon decodability: {acc:.3f} (chance={chance:.3f}, lift={acc/chance:.2f}x)")
            else:
                print("[Q/K] Capture failed")
            
            qkv.remove()
        
        print()
    
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_fixed_mi(
        model_dir=Path("data/models/Olmo-3-7B-Instruct"),
        atlas_dir=Path("data/atlas"),
    )