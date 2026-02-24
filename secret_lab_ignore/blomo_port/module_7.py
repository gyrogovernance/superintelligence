"""
Module 7: The Suffix Residual Router Port

Goal:
Bolmo relies on an expensive Subword Trie search to enrich bytes with suffix embeddings.
We replace this entirely with the Router Kernel.

We map Router observables (State Index, Horizon, Vertex, Phase) to the mode Suffix ID.
During generation, we patch the embedding layer to ignore the Trie search.
Instead, it steps the Router Kernel, looks up the embedding ID from our Finite State Table,
and applies the residual.

If this maintains coherence, we prove the Router's physics natively track
subword semantics, unlocking the removal of the Trie search.

Extraction phase uses the suffix automaton (Step 2) when available for fast
expand_byte_ids, falling back to tokenizer.expand_byte_ids otherwise.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches, token_to_byte_and_fused
from module_0_baseline import baseline_generate
from src.router.kernel import RouterKernel

ATLAS_DIR = PROJECT_ROOT / "data" / "atlas"
SUFFIX_AUTOMATON_PATH = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "suffix_automaton.npz"


def _load_suffix_automaton():  # -> Optional[SuffixAutomaton], avoid import at top
    """Load suffix automaton from cache if present. Build with adaptors/suffix_adaptor.py first."""
    if not SUFFIX_AUTOMATON_PATH.exists():
        return None
    try:
        from adaptors.suffix_adaptor import SuffixAutomaton
        return SuffixAutomaton.load(SUFFIX_AUTOMATON_PATH)
    except Exception:
        return None


@dataclass
class SuffixBackoffTables:
    state_to_eid: np.ndarray        # [65536]
    hvp_to_eid: np.ndarray          # [256*16] (Horizon * 16 + Vertex * 4 + Phase)
    horizon_to_eid: np.ndarray      # [256]
    global_default_eid: int

@torch.inference_mode()
def extract_router_to_suffix_mapping(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 200,
) -> SuffixBackoffTables:
    """Runs baseline generation to learn which Router states map to which Suffix IDs."""
    bolmo_reset_local_caches(model)
    kernel = RouterKernel(ATLAS_DIR, batch_size=1)
    kernel.reset()
    torch.manual_seed(42)

    device = next(model.parameters()).device
    off = int(getattr(tokenizer, "offset", 4))

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    L0 = input_ids.shape[1]

    # Generate to get the "ground truth" subword expansions
    out = model.generate(input_ids, max_length=L0 + max_new_tokens, do_sample=True, temperature=1.0, top_k=0)
    full_ids = out[0].tolist()

    automaton = _load_suffix_automaton()
    if automaton is not None:
        from adaptors.suffix_adaptor import expand_byte_ids_with_automaton  # noqa: PLC0415
        expanded_ids = expand_byte_ids_with_automaton(full_ids, automaton, tokenizer)
    else:
        expanded_ids = tokenizer.expand_byte_ids(full_ids)

    # Track mappings
    state_counts = defaultdict(Counter)
    hvp_counts = defaultdict(Counter)
    horizon_counts = defaultdict(Counter)
    global_counts = Counter()

    for pos, tid in enumerate(full_ids):
        b, _ = token_to_byte_and_fused(int(tid), off)
        if b is None: continue
        
        bv = int(b) & 0xFF
        kernel.step_byte(bv)

        sidx = int(kernel.state_index[0])
        h = int(kernel.current_horizon[0])
        v = int(kernel.current_vertex[0])
        p = int(kernel.current_phase[0])
        
        eid = int(expanded_ids[pos]) if pos < len(expanded_ids) else 0

        state_counts[sidx][eid] += 1
        hvp_counts[h * 16 + v * 4 + p][eid] += 1
        horizon_counts[h][eid] += 1
        global_counts[eid] += 1

    # Compile mode tables (most frequent ID for each state)
    def to_array(size, counts):
        arr = np.full(size, -1, dtype=np.int32)
        for k, c in counts.items():
            arr[k] = c.most_common(1)[0][0]
        return arr

    return SuffixBackoffTables(
        state_to_eid=to_array(65536, state_counts),
        hvp_to_eid=to_array(256 * 16, hvp_counts),
        horizon_to_eid=to_array(256, horizon_counts),
        global_default_eid=global_counts.most_common(1)[0][0]
    )

class RouterSuffixPatch:
    """Replaces Bolmo's Subword Trie Residual with Router Physics Backoff Lookup."""
    def __init__(self, model: Any, tokenizer: Any, tables: SuffixBackoffTables, blend: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.tables = tables
        self.blend = float(blend)
        self._orig_embed = None
        self._rk = None

    def __enter__(self):
        le = self.model.model.local_encoder
        self._orig_embed = le._embed
        off = int(getattr(self.tokenizer, "offset", 4))
        
        self._rk = RouterKernel(ATLAS_DIR, batch_size=1)
        self._rk.reset()
        rk = self._rk 
        tables = self.tables
        blend = self.blend

        def _patched_embed(tokens: torch.Tensor, expanded_input_ids: Optional[torch.Tensor] = None):
            byte_emb = le.byte_embedding(tokens)
            
            # The ground truth suffix from the expensive Trie search
            orig_suffix = le.subword_embedding(expanded_input_ids) if expanded_input_ids is not None else torch.zeros_like(byte_emb)

            B, L = tokens.shape
            chosen_eids = np.full(L, tables.global_default_eid, dtype=np.int32)

            # Step the Router and lookup the prediction
            for t in range(L):
                tid = int(tokens[0, t].item())
                b, _ = token_to_byte_and_fused(tid, off)
                if b is None: continue

                bv = int(b) & 0xFF
                rk.step_byte(bv)

                sidx = int(rk.state_index[0])
                h = int(rk.current_horizon[0])
                v = int(rk.current_vertex[0])
                p = int(rk.current_phase[0])

                if tables.state_to_eid[sidx] >= 0:
                    chosen_eids[t] = tables.state_to_eid[sidx]
                elif tables.hvp_to_eid[h * 16 + v * 4 + p] >= 0:
                    chosen_eids[t] = tables.hvp_to_eid[h * 16 + v * 4 + p]
                elif tables.horizon_to_eid[h] >= 0:
                    chosen_eids[t] = tables.horizon_to_eid[h]

            # Fetch our predicted embeddings
            eid_t = torch.from_numpy(chosen_eids).to(device=tokens.device, dtype=torch.long).unsqueeze(0)
            router_suffix = le.subword_embedding(eid_t)
            
            # Blend: 1.0 means PURE ROUTER (ignoring the Trie entirely)
            suffix = (1.0 - blend) * orig_suffix + blend * router_suffix
            return byte_emb + suffix

        le._embed = _patched_embed
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_embed is not None:
            self.model.model.local_encoder._embed = self._orig_embed

def run_module_7(
    model: Any,
    tokenizer: Any,
    prompt: str = "Language modeling is ",
    max_new_tokens: int = 300,
) -> None:
    print("\n[Module 7] Suffix Residual Router Port")
    print("Goal: Can the Router State substitute for the Subword Trie search?")
    
    # 1. Baseline Extraction
    print("\n--- Phase A: Extracting Subword semantic mapping to Router States ---")
    t0 = time.perf_counter()
    tables = extract_router_to_suffix_mapping(model, tokenizer, prompt, max_new_tokens)
    print(f"Extraction complete in {time.perf_counter()-t0:.2f}s")
    
    # 2. True Baseline Generate
    bolmo_reset_local_caches(model)
    torch.manual_seed(42)
    base_text, base_ids, dt_base = baseline_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=0)
    print(f"\n[BASELINE (Uses expensive Trie Search)] tokens={len(base_ids)} time={dt_base:.2f}s")
    print("  " + base_text[:200].replace("\n", "\\n"))
    
    # 3. Blends
    for blend in [0.0, 0.5, 1.0]:
        bolmo_reset_local_caches(model)
        torch.manual_seed(42)
        
        mode_name = "PURE ROUTER" if blend == 1.0 else f"Blend {blend}"
        
        t0 = time.perf_counter()
        with RouterSuffixPatch(model, tokenizer, tables, blend=blend):
            pat_text, pat_ids, dt_pat = baseline_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=0)
        
        n = min(len(base_ids), len(pat_ids))
        eq = sum(1 for i in range(n) if base_ids[i] == pat_ids[i])
        
        print(f"\n[{mode_name}] tokens={len(pat_ids)} time={dt_pat:.2f}s match={eq}/{n}={eq/max(1,n):.3f}")
        print("  " + pat_text[:200].replace("\n", "\\n"))

    print("\n[Module 7] Done.")