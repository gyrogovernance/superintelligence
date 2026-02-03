"""
GyroSelect: Kernel-guided candidate selection for aligned generation.

Instead of biasing logits or generating standalone, we let OLMo propose
top-k candidates and use CGM kernel properties to select among them.

This keeps language coherence (from OLMo) while applying alignment
principles (from CGM) as a selection criterion.

Run:
  python research_mechanistic_interpretability/gyroselect.py
"""

from __future__ import annotations

import os
import sys
import time
import gc
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

class Config:
    model_path = Path("data/models/Olmo-3-7B-Instruct")
    atlas_dir = Path("data/atlas")
    codec_path = Path("data/atlas/semantic_codec.npz")
    
    # Generation parameters
    top_k: int = 32                    # Candidates from OLMo to consider
    max_new_tokens: int = 50           # Tokens to generate
    temperature: float = 0.8           # OLMo sampling temperature
    
    # Kernel scoring weights
    horizon_smoothness: float = 0.3    # Prefer smooth horizon transitions
    vertex_coherence: float = 0.4      # Prefer K₄-coherent transitions
    weight_penalty: float = 0.2        # Prefer lighter mask weights
    phase_bonus: float = 0.1           # Prefer phase-preserving transitions
    
    # Runtime
    device: str = "cpu"
    dtype = torch.bfloat16


CFG = Config()


# =============================================================================
# Kernel Scorer
# =============================================================================

class KernelScorer:
    """
    Scores token candidates based on CGM kernel properties.
    
    The key insight: we're not generating FROM the kernel, we're using
    the kernel to EVALUATE candidates that OLMo proposes.
    """
    
    def __init__(self, atlas_dir: Path, codec: Any):
        from src.router.kernel import RouterKernel
        
        self.kernel = RouterKernel(atlas_dir)
        self.codec = codec
        
        # Load atlas data for scoring
        with np.load(atlas_dir / "phenomenology.npz") as z:
            self.byte_weight = z["byte_weight"].astype(np.float32) / 12.0
            self.byte_charge = z["byte_charge"].astype(np.int64)
            self.state_horizon = z["state_horizon"]
            self.state_vertex = z["state_vertex"]
        
        self.epistemology = np.load(atlas_dir / "epistemology.npy", mmap_mode="r")
    
    def reset(self) -> None:
        self.kernel.reset()
    
    def score_candidates(
        self,
        candidate_ids: List[int],
        cfg: Config,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Score each candidate token based on kernel properties.
        
        Returns:
            scores: array of scores for each candidate
            details: list of scoring details for analysis
        """
        current_idx = self.kernel.state_index
        current_h = int(self.state_horizon[current_idx])
        current_chi = int(self.state_vertex[current_idx])
        
        scores = np.zeros(len(candidate_ids), dtype=np.float32)
        details = []
        
        for i, token_id in enumerate(candidate_ids):
            # Encode token to 4 bytes
            try:
                bs = self.codec.encode(int(token_id))
            except:
                scores[i] = -100.0
                details.append({"error": "encode_failed"})
                continue
            
            # Simulate trajectory through these 4 bytes
            idx = current_idx
            total_weight = 0.0
            horizon_changes = 0
            vertex_changes = 0
            
            prev_h = current_h
            prev_chi = current_chi
            
            for b in bs:
                b = int(b)
                next_idx = int(self.epistemology[idx, b])
                next_h = int(self.state_horizon[next_idx])
                next_chi = int(self.state_vertex[next_idx])
                
                total_weight += self.byte_weight[b]
                if next_h != prev_h:
                    horizon_changes += 1
                if next_chi != prev_chi:
                    vertex_changes += 1
                
                idx = next_idx
                prev_h = next_h
                prev_chi = next_chi
            
            # Compute score components
            # 1. Horizon smoothness: fewer horizon changes = better
            h_score = 1.0 - (horizon_changes / 4.0)
            
            # 2. Vertex coherence: staying in same or adjacent K₄ vertex
            # XOR of 3 means opposite vertex (bad), 0 means same (good), 1 or 2 means adjacent
            final_chi = int(self.state_vertex[idx])
            chi_diff = current_chi ^ final_chi
            if chi_diff == 0:
                v_score = 1.0   # Same vertex
            elif chi_diff == 3:
                v_score = -0.5  # Opposite vertex
            else:
                v_score = 0.5   # Adjacent vertex
            
            # 3. Weight penalty: prefer lighter masks
            w_score = 1.0 - (total_weight / 4.0)
            
            # 4. Phase bonus: reward returning to similar state structure
            final_h = int(self.state_horizon[idx])
            # Simple phase score: how close is final horizon to current?
            h_dist = min(abs(final_h - current_h), 256 - abs(final_h - current_h))
            p_score = 1.0 - (h_dist / 128.0)
            
            # Combine scores
            score = (
                cfg.horizon_smoothness * h_score +
                cfg.vertex_coherence * v_score +
                cfg.weight_penalty * w_score +
                cfg.phase_bonus * p_score
            )
            
            scores[i] = score
            details.append({
                "token_id": token_id,
                "h_score": h_score,
                "v_score": v_score,
                "w_score": w_score,
                "p_score": p_score,
                "total": score,
                "horizon_changes": horizon_changes,
                "vertex_changes": vertex_changes,
                "chi_transition": f"{current_chi}->{final_chi}",
            })
        
        return scores, details
    
    def commit_token(self, token_id: int) -> None:
        """Step the kernel through a token's bytes (after selection)."""
        bs = self.codec.encode(int(token_id))
        for b in bs:
            self.kernel.step_byte(int(b))


# =============================================================================
# Generator
# =============================================================================

class GyroSelectGenerator:
    """
    Generates text using OLMo for proposals and kernel for selection.
    """
    
    def __init__(self, model: Any, tokenizer: Any, scorer: KernelScorer, cfg: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.cfg = cfg
        self.device = torch.device(cfg.device)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[str, List[dict]]:
        """
        Generate text using kernel-guided selection.
        
        Returns:
            generated_text: the complete generated text
            selection_log: details of each selection decision
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        
        # Encode prompt
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        
        # Initialize kernel at prompt
        self.scorer.reset()
        for tid in input_ids[0].tolist():
            self.scorer.commit_token(tid)
        
        generated_ids = input_ids[0].tolist()
        selection_log = []
        
        if verbose:
            print(f"\nGenerating {max_tokens} tokens...")
            print(f"Prompt: {prompt[:80]}...")
            print("-" * 60)
        
        for step in range(max_tokens):
            # Get OLMo's next-token distribution
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=torch.tensor([generated_ids], device=self.device)
                )
                logits = outputs.logits[0, -1, :].float()
            
            # Apply temperature and get top-k candidates
            probs = F.softmax(logits / self.cfg.temperature, dim=-1)
            top_probs, top_indices = probs.topk(self.cfg.top_k)
            
            candidates = top_indices.tolist()
            olmo_probs = top_probs.tolist()
            
            # Score candidates with kernel
            kernel_scores, details = self.scorer.score_candidates(candidates, self.cfg)
            
            # Combine OLMo probability with kernel score
            # Normalize kernel scores to [0, 1]
            k_min, k_max = kernel_scores.min(), kernel_scores.max()
            if k_max > k_min:
                kernel_norm = (kernel_scores - k_min) / (k_max - k_min)
            else:
                kernel_norm = np.ones_like(kernel_scores) * 0.5
            
            # Combined score: geometric mean of OLMo prob and kernel score
            combined = np.sqrt(np.array(olmo_probs) * (kernel_norm + 0.1))
            
            # Select best
            best_idx = int(np.argmax(combined))
            selected_id = candidates[best_idx]
            
            # Log selection
            selection_log.append({
                "step": step,
                "selected_id": selected_id,
                "selected_token": self.tokenizer.decode([selected_id]),
                "olmo_rank": best_idx,
                "olmo_prob": olmo_probs[best_idx],
                "kernel_score": float(kernel_scores[best_idx]),
                "combined_score": float(combined[best_idx]),
                "details": details[best_idx],
            })
            
            # Commit to kernel and add to sequence
            self.scorer.commit_token(selected_id)
            generated_ids.append(selected_id)
            
            if verbose and step % 10 == 0:
                print(f"  Step {step}: selected '{self.tokenizer.decode([selected_id])}' "
                      f"(rank {best_idx}, kernel={kernel_scores[best_idx]:.3f})")
        
        # Decode full output
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return output_text, selection_log


# =============================================================================
# Analysis
# =============================================================================

def analyze_selections(log: List[dict]) -> dict:
    """Analyze selection patterns."""
    if not log:
        return {}
    
    ranks = [entry["olmo_rank"] for entry in log]
    kernel_scores = [entry["kernel_score"] for entry in log]
    
    # How often did kernel change OLMo's top choice?
    rank_0_count = sum(1 for r in ranks if r == 0)
    intervention_rate = 1.0 - (rank_0_count / len(ranks))
    
    # Average kernel score
    avg_kernel = np.mean(kernel_scores)
    
    # K₄ transition patterns
    chi_transitions = [entry["details"].get("chi_transition", "?") for entry in log]
    same_vertex = sum(1 for t in chi_transitions if t.split("->")[0] == t.split("->")[1])
    
    return {
        "total_tokens": len(log),
        "intervention_rate": intervention_rate,
        "avg_kernel_score": avg_kernel,
        "avg_olmo_rank": np.mean(ranks),
        "same_vertex_rate": same_vertex / len(log),
        "rank_distribution": {
            "rank_0": rank_0_count,
            "rank_1-5": sum(1 for r in ranks if 1 <= r <= 5),
            "rank_6+": sum(1 for r in ranks if r >= 6),
        }
    }


# =============================================================================
# Main
# =============================================================================

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.agent.adapters import SemanticTokenCodec
    
    torch.set_num_threads(12)
    os.environ["OMP_NUM_THREADS"] = "12"
    
    print("=" * 70)
    print("GyroSelect: Kernel-Guided Candidate Selection")
    print("=" * 70)
    print(f"Device: {CFG.device}, Top-k: {CFG.top_k}, Max tokens: {CFG.max_new_tokens}")
    sys.stdout.flush()
    
    # Check memory
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        print(f"Available RAM: {avail:.1f} GB")
        if avail < 16:
            print("WARNING: Low memory, model load may fail")
    except ImportError:
        pass
    
    # Load components
    print("\n[1] Loading model...")
    gc.collect()
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    print("    Tokenizer OK")
    
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_path,
        torch_dtype=CFG.dtype,
        device_map={"": CFG.device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("    Model OK")
    
    print("\n[2] Loading kernel and codec...")
    codec = SemanticTokenCodec.load(CFG.codec_path)
    scorer = KernelScorer(CFG.atlas_dir, codec)
    print("    Scorer OK")
    
    generator = GyroSelectGenerator(model, tokenizer, scorer, CFG)
    
    # Test prompts
    prompts = [
        "The principles of good governance require",
        "Artificial intelligence systems must be designed to",
        "The relationship between structure and function in",
    ]
    
    print("\n" + "=" * 70)
    print("GENERATION")
    print("=" * 70)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"PROMPT {i}/{len(prompts)}")
        print(f"{'='*70}")
        
        output, log = generator.generate(prompt, verbose=True)
        stats = analyze_selections(log)
        
        print(f"\n--- OUTPUT ---")
        print(output)
        
        print(f"\n--- ANALYSIS ---")
        print(f"Intervention rate: {stats['intervention_rate']:.1%}")
        print(f"  (How often kernel chose different from OLMo's top-1)")
        print(f"Avg kernel score: {stats['avg_kernel_score']:.3f}")
        print(f"Avg OLMo rank of selected: {stats['avg_olmo_rank']:.1f}")
        print(f"Same K₄ vertex rate: {stats['same_vertex_rate']:.1%}")
        print(f"Rank distribution: {stats['rank_distribution']}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()