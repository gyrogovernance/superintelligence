# research_mechanistic_interpretability/cgm_phase_diagram.py
"""
CGM Holographic Phase Diagram.

Purpose:
  Map the macroscopic evolution of the Transformer's state through the
  lens of the Router Kernel. We are not testing *if* structure exists,
  but *measuring* the flow of the system through the CGM coordinate space.

Measurements:
  1. Global Hodge Aperture (Layer-wise): How 'Cycle-heavy' is the residual stream?
  2. K4 Spectral Stiffness: Does the representation maintain the 1+3 tetrahedral split?
  3. MLP Monodromy: Does the MLP act as a rotation on the K4 vertices?
  4. Horizon Diffusion: Does the model narrow or widen the horizon distribution?

This creates a 'medical chart' of the model's geometric health per layer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.adapters import SemanticTokenCodec
from src.router.kernel import RouterKernel
# -----
# Constants & Geometry
# -----

B_K4 = np.array(
    [
        [-1, -1, -1, 0, 0, 0],
        [1, 0, 0, -1, -1, 0],
        [0, 1, 0, 1, 0, -1],
        [0, 0, 1, 0, 1, 1],
    ],
    dtype=np.float64,
)
# Exact projectors
P_GRAD = 0.25 * (B_K4.T @ B_K4)
P_CYCLE = np.eye(6, dtype=np.float64) - P_GRAD

# -----
# Microscope Setup
# -----

@dataclass
class MicroscopeLabels:
    vertices: np.ndarray  # (N,)
    horizons: np.ndarray  # (N,)

def prepare_microscope(
    model_dir: Path, atlas_dir: Path, n_tokens: int = 2048, seed: int = 42
) -> tuple[Any, Any, MicroscopeLabels, torch.Tensor]:
    """
    Load model, codec, and prepare a balanced set of tokens labeled by the Kernel.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[SETUP] Loading Model & Atlas...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    codec = SemanticTokenCodec.load(model_dir / "gyro_codebook.npz")
    
    # Sample and Label
    print("[SETUP] Sampling Tokens...")
    rng = np.random.default_rng(seed)
    all_tokens = np.arange(int(model.config.vocab_size))
    # Oversample to allow balancing
    raw_tokens = rng.choice(all_tokens, size=min(n_tokens * 2, len(all_tokens)), replace=False)
    
    K = RouterKernel(atlas_dir)
    vertices = []
    horizons = []
    
    for tid in raw_tokens:
        K.reset()
        bs = codec.encode(int(tid))
        for b in bs: K.step_byte(b)
        vertices.append(K.current_vertex)
        horizons.append(K.current_horizon)
        
    vertices = np.array(vertices, dtype=np.int32)
    horizons = np.array(horizons, dtype=np.int32)
    
    # Balance Vertices (essential for spectral tests)
    counts = np.bincount(vertices, minlength=4)
    min_count = counts.min()
    balanced_idxs = []
    for v in range(4):
        v_idxs = np.where(vertices == v)[0]
        balanced_idxs.extend(v_idxs[:min_count])
    
    balanced_idxs = np.array(balanced_idxs)
    rng.shuffle(balanced_idxs)
    
    final_tokens = raw_tokens[balanced_idxs]
    final_labels = MicroscopeLabels(
        vertices=vertices[balanced_idxs],
        horizons=horizons[balanced_idxs]
    )
    
    print(f"[SETUP] Prepared {len(final_tokens)} tokens (Balanced K4).")
    
    # Extract Embeddings once
    with torch.no_grad():
        E = model.model.embed_tokens.weight[final_tokens].detach()
    
    return model, tokenizer, final_labels, E

# -----
# Metric: Global Hodge Aperture
# -----

def measure_layer_aperture(
    hidden_states: torch.Tensor, labels: MicroscopeLabels
) -> dict[str, float]:
    """
    Project the hidden state interaction matrix onto the K4 graph 
    and measure the Cycle fraction (Aperture).
    
    Logic:
    1. Compute centroids of hidden states for each vertex 0..3.
    2. Compute the 4x4 Interaction Matrix M = Centroids @ Centroids.T.
    3. Extract antisymmetric flow F_ij = M_ij - M_ji.
    4. Project F onto Cycle space.
    """
    X = hidden_states
    
    # Compute Centroids
    centroids = []
    for v in range(4):
        mask = (labels.vertices == v)
        if mask.sum() > 0:
            c = X[mask].mean(dim=0)
        else:
            c = torch.zeros(X.shape[1], device=X.device)
        centroids.append(c)
    C = torch.stack(centroids) # (4, D)
    
    # Interaction Matrix (Transport)
    M = (C @ C.T).cpu().numpy()
    
    # Extract Edge Flow (Antisymmetric component)
    # Edge mapping: 0:0-1, 1:0-2, 2:0-3, 3:1-2, 4:1-3, 5:2-3
    y = np.zeros(6)
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for idx, (i,j) in enumerate(pairs):
        y[idx] = M[i,j] - M[j,i]
        
    # Energy
    total_energy = np.sum(y**2) + 1e-12
    cycle_flow = P_CYCLE @ y
    cycle_energy = np.sum(cycle_flow**2)
    
    aperture = cycle_energy / total_energy
    
    # Also measure Diagonal Dominance (Stasis vs Transport)
    diag_energy = np.trace(M)
    off_diag_energy = np.sum(M) - diag_energy
    stasis_ratio = diag_energy / (off_diag_energy + 1e-12)
    
    return {
        "aperture": aperture,
        "stasis": stasis_ratio,
        "total_flow": total_energy
    }

# -----
# Metric: K4 Spectral Stiffness
# -----

def measure_spectral_stiffness(hidden_states: torch.Tensor, labels: MicroscopeLabels) -> float:
    """
    Check if the representation maintains the 1+3 degeneracy of K4.
    We look at the eigenvalues of the Symmetric Interaction Matrix (Sym = M + M.T).
    Ideal K4 (Tetrahedron) should have lambda_2 approx equal to lambda_3 approx equal to lambda_4.
    
    Returns: Degeneracy score (std dev of the bottom 3 eigenvalues). Lower is better.
    """
    X = hidden_states
    centroids = []
    for v in range(4):
        mask = (labels.vertices == v)
        if mask.sum() > 0:
            c = X[mask].mean(dim=0)
        else:
            c = torch.zeros(X.shape[1], device=X.device)
        centroids.append(c)
    C = torch.stack(centroids)
    
    M = (C @ C.T).cpu().numpy()
    S = 0.5 * (M + M.T)
    
    eigs = np.linalg.eigvalsh(S)
    # Sort descending
    eigs = np.sort(eigs)[::-1]
    
    # The first eigenvalue captures the "Common Mode" (Grand Mean).
    # The next three capture the spatial structure.
    spatial_modes = eigs[1:]
    
    # Normalize by scale
    scale = np.abs(np.mean(spatial_modes)) + 1e-12
    stiffness = np.std(spatial_modes) / scale
    
    return stiffness

# -----
# Metric: MLP Monodromy
# -----

def measure_mlp_monodromy(
    model: Any, layer_idx: int, hidden_states: torch.Tensor, labels: MicroscopeLabels
) -> float:
    """
    Does the MLP rotate the vertex structure?
    We compare the Vertex Centroids BEFORE and AFTER the MLP.
    
    Monodromy = 1 - CosineSimilarity(Before_Centroids, After_Centroids)
    High Monodromy => The MLP is twisting the space.
    Low Monodromy => The MLP is acting scaler/elementwise without rotation.
    """
    layer = model.model.layers[layer_idx]
    
    # We need to run the MLP.
    # Note: OLMo MLP is gate_proj * up_proj -> down_proj
    # We can just use the module if we reshape inputs
    # hidden_states: (N, D)
    
    # Move to same device/type
    X = hidden_states.to(layer.mlp.gate_proj.weight.device).to(layer.mlp.gate_proj.weight.dtype)
    
    with torch.no_grad():
        gate = layer.mlp.gate_proj(X)
        up = layer.mlp.up_proj(X)
        inter = torch.nn.functional.silu(gate) * up
        Y = layer.mlp.down_proj(inter) # Output of MLP
    
    # Compute centroids for Input X and Output Y
    # We flatten the 4 centroids into a single 4D vector for correlation
    
    c_in_list = []
    c_out_list = []
    
    for v in range(4):
        mask = (labels.vertices == v)
        if mask.sum() > 0:
            c_in_list.append(X[mask].mean(dim=0))
            c_out_list.append(Y[mask].mean(dim=0))
        else:
            z = torch.zeros(X.shape[1], device=X.device)
            c_in_list.append(z)
            c_out_list.append(z)
            
    # Stack
    C_in = torch.stack(c_in_list).flatten()
    C_out = torch.stack(c_out_list).flatten()
    
    cos = torch.nn.functional.cosine_similarity(C_in.unsqueeze(0), C_out.unsqueeze(0)).item()
    
    return 1.0 - cos

# -----
# Main Execution
# -----

def main():
    print("CGM Holographic Phase Diagram")
    print("-----------------------------")
    t0 = time.time()
    
    MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
    ATLAS_DIR = Path("data/atlas")
    
    model, _tokenizer, labels, embeddings = prepare_microscope(MODEL_DIR, ATLAS_DIR)
    
    n_layers = model.config.num_hidden_layers
    
    print(f"\n[PHASE MAP] Scanning {n_layers} layers...")
    print(f"{'L':<3} | {'Aperture':<8} | {'Stiffness':<9} | {'Monodromy':<9} | {'Stasis':<8}")
    print("-" * 45)
    
    # We will "push" the embeddings through the network layer by layer
    # to simulate the residual stream evolution without full forward passes
    # This assumes we can treat the residual stream as the state.
    # Note: For exactness in transformers, we should technically run the full model,
    # but running layer-by-layer on the initial embeddings + accumulation is a valid
    # "Tangent Space" approximation for structural tomography. 
    # **Correction**: To be precise, we will run a forward pass and capture hidden states.
    
    # Run full forward pass to get valid hidden states
    print("... Running forward pass to capture states ...")
    with torch.no_grad():
        outputs = model(
            inputs_embeds=embeddings.unsqueeze(0), 
            output_hidden_states=True
        )
    
    # Iterate layers
    # hidden_states[0] is embeddings
    # hidden_states[i] is output of layer i-1 (input to layer i)
    
    history = []
    
    for i in range(n_layers + 1):
        h_state = outputs.hidden_states[i].squeeze(0) # (N, D)
        
        # 1. Aperture & Stasis
        ap_metrics = measure_layer_aperture(h_state, labels)
        
        # 2. Stiffness
        stiffness = measure_spectral_stiffness(h_state, labels)
        
        # 3. Monodromy (only valid for input to a layer, so skip last)
        monodromy = 0.0
        if i < n_layers:
            monodromy = measure_mlp_monodromy(model, i, h_state, labels)
            
        print(f"{i:<3} | {ap_metrics['aperture']:.4f}   | {stiffness:.4f}    | {monodromy:.4f}    | {ap_metrics['stasis']:.2f}")
        
        history.append({
            "layer": i,
            "aperture": ap_metrics['aperture'],
            "stiffness": stiffness,
            "monodromy": monodromy
        })

    print("-" * 45)
    
    # Analysis
    print("\n[DIAGNOSIS]")
    
    # Check for Aperture convergence
    final_ap = history[-1]['aperture']
    target_ap = 0.0207
    print(f"Final Aperture: {final_ap:.4f} (Target ~ {target_ap})")
    if abs(final_ap - target_ap) < 0.01:
        print(">> SYSTEM CONVERGED TO CGM TARGET")
    elif final_ap < target_ap:
        print(">> SYSTEM OVER-DAMPED (Too much Gradient)")
    else:
        print(">> SYSTEM UNDER-DAMPED (Too much Cycle)")
        
    # Check for Stiffness (K4 integrity)
    avg_stiff = np.mean([h['stiffness'] for h in history])
    print(f"Avg Spectral Stiffness: {avg_stiff:.4f} (Lower is more Tetrahedral)")
    
    # Check for Monodromy Spike
    max_mono = max([h['monodromy'] for h in history[:-1]])
    print(f"Max MLP Twist: {max_mono:.4f}")
    
    print(f"\nTotal dt={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()