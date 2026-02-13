from __future__ import annotations

import os, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, os.getcwd())
from src.router.kernel import RouterKernel

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ATLAS_DIR = Path("data/atlas")

# ---------------- knobs (edit here; no flags) ----------------
SEED = 0
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.7
TOP_K = 40

PROBE_LAYER = 31  # full-attn layer: 3,7,11,15,19,23,27,31

PROMPTS = [
    "The purpose of good governance is",
    "Mathematics reveals that",
    "In three dimensions, the structure",
    "The relationship between energy and matter",
]

# theory-informed "information set" guard sizes
M_GUARD_LIST = [0, 4, 8, 16]  # 0 means wedge-only; 8 is the canonical threshold

# choose wedges by mass target or fixed topN
USE_MASS_TARGET = False
MASS_TARGET = 0.95  # if USE_MASS_TARGET, keep min wedges whose mass >= this
TOPN_WEDGES = 2     # else keep topN wedges
# ------------------------------------------------------------

D_MODEL = 4096
N_BOUNDARY = 256
N_FIBER = 16


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_eager() -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()
    return model, tok


def get_layer_kv_from_cache(past_key_values, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    pkv = past_key_values

    if isinstance(pkv, (tuple, list)):
        return pkv[layer]

    if hasattr(pkv, "layers"):
        L = pkv.layers[layer]
        return L.keys, L.values

    if hasattr(pkv, "to_legacy_cache"):
        legacy = pkv.to_legacy_cache()
        return legacy[layer]

    raise TypeError(f"Unsupported cache type: {type(pkv)}")


def embed_to_byte(vec4096: torch.Tensor) -> int:
    with torch.no_grad():
        x = vec4096.float().view(N_BOUNDARY, N_FIBER)
        e = (x * x).sum(dim=1)
        h = int(torch.argmax(e).item())
        v = x[h, :]
        bits = 0
        for i in range(8):
            if v[i * 2].item() > 0:
                bits |= (1 << i)
        return bits & 0xFF


def sample_topk(logits: torch.Tensor) -> int:
    logits = logits / float(TEMPERATURE)
    topv, topi = torch.topk(logits, k=int(TOP_K))
    p = torch.softmax(topv, dim=-1)
    idx = int(torch.multinomial(p, 1).item())
    return int(topi[idx].item())


class WedgeLabeler:
    def __init__(self, kernel: RouterKernel, embed_weight: torch.Tensor):
        self.kernel = kernel
        self.embed = embed_weight
        self.chi_hist: List[int] = []

    def consume_token(self, token_id: int) -> None:
        b = embed_to_byte(self.embed[int(token_id)])
        self.kernel.step_byte(b)
        self.chi_hist.append(int(self.kernel.current_vertex) & 3)


def wedge_masses_past_only(
    attn_heads_k: torch.Tensor,  # [heads,k]
    chi_hist: List[int],
    key_start: int,
    abs_pos_count: int,
) -> Tuple[np.ndarray, float]:
    A = attn_heads_k.float().mean(dim=0)  # [k]
    k_len = int(A.shape[0])

    abs_self = abs_pos_count - 1
    j_self = abs_self - key_start
    self_mass = float(A[j_self].item()) if 0 <= j_self < k_len else 0.0

    m = np.zeros(4, dtype=np.float64)
    for j in range(k_len):
        abs_j = key_start + j
        if abs_j == abs_self:
            continue
        if 0 <= abs_j < len(chi_hist):
            v = int(chi_hist[abs_j]) & 3
            m[v] += float(A[j].item())

    return m, self_mass


def choose_wedges(m: np.ndarray) -> set[int]:
    order = np.argsort(m)[::-1]
    if not USE_MASS_TARGET:
        return set(int(x) for x in order[: int(TOPN_WEDGES)])
    total = float(m.sum()) + 1e-18
    acc = 0.0
    keep: set[int] = set()
    for idx in order:
        keep.add(int(idx))
        acc += float(m[int(idx)])
        if acc / total >= float(MASS_TARGET):
            break
    return keep


def build_mask_top_wedges_plus_infoset(
    *,
    attn_heads_k: torch.Tensor,   # [heads,k]
    chi_hist: List[int],
    key_start: int,
    abs_pos_count: int,
    keep_wedges: set[int],
    M_guard: int,
) -> torch.Tensor:
    """
    Keep:
      - self key
      - all keys in keep_wedges
      - plus top-M_guard keys (by attention weight) among the excluded keys
    """
    A = attn_heads_k.float().mean(dim=0)  # [k]
    k_len = int(A.shape[0])

    abs_self = abs_pos_count - 1
    j_self = abs_self - key_start

    mask = torch.zeros(k_len, dtype=torch.bool)

    # Always keep self if present
    if 0 <= j_self < k_len:
        mask[j_self] = True

    # First pass: mark wedge-kept keys; collect candidates outside wedges
    outside_idxs: List[int] = []
    outside_scores: List[float] = []

    for j in range(k_len):
        if j == j_self:
            continue
        abs_j = key_start + j
        if 0 <= abs_j < len(chi_hist):
            v = int(chi_hist[abs_j]) & 3
            if v in keep_wedges:
                mask[j] = True
            else:
                outside_idxs.append(j)
                outside_scores.append(float(A[j].item()))
        else:
            # unlabeled (should be rare after prompt consumption); keep conservatively
            mask[j] = True

    if M_guard > 0 and outside_idxs:
        scores = np.array(outside_scores, dtype=np.float64)
        idxs = np.array(outside_idxs, dtype=np.int64)
        M = min(int(M_guard), int(idxs.shape[0]))
        keep_extra = idxs[np.argpartition(scores, -M)[-M:]]
        mask[torch.tensor(keep_extra, dtype=torch.long)] = True

    return mask


def oracle_context_error(
    attn_heads_k: torch.Tensor,  # [heads,k]
    V_bhkd: torch.Tensor,        # [1,heads,k,dh]
    mask_k: torch.Tensor,        # [k] bool
) -> Tuple[float, float]:
    w = attn_heads_k.float()
    V = V_bhkd.float()[0]  # [h,k,dh]

    ctx_full = torch.einsum("hk,hkd->hd", w, V)

    mk = mask_k.to(w.device)
    w2 = w * mk.unsqueeze(0)
    w2 = w2 / (w2.sum(dim=-1, keepdim=True) + 1e-12)
    ctx_pr = torch.einsum("hk,hkd->hd", w2, V)

    a = ctx_full.reshape(-1)
    b = ctx_pr.reshape(-1)

    cos = float(F.cosine_similarity(a, b, dim=0).item())
    rel = float(torch.norm(b - a).item() / (torch.norm(a).item() + 1e-12))
    return cos, rel


@torch.inference_mode()
def run_prompt(model: AutoModelForCausalLM, tok: PreTrainedTokenizerBase, prompt: str) -> None:
    device = next(model.parameters()).device
    embed = model.model.embed_tokens.weight

    kernel = RouterKernel(atlas_dir=ATLAS_DIR)
    labeler = WedgeLabeler(kernel, embed)

    enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = enc.input_ids[0].tolist()
    if not prompt_ids:
        print("Empty prompt")
        return

    past = None
    logits_last: Optional[torch.Tensor] = None

    # consume prompt token-by-token
    for tid in prompt_ids:
        x = torch.tensor([[int(tid)]], device=device, dtype=torch.long)
        out = model(x, use_cache=True, past_key_values=past, output_attentions=False, return_dict=True)
        past = out.past_key_values
        logits_last = out.logits[0, -1, :]
        labeler.consume_token(int(tid))

    assert logits_last is not None
    next_tok = sample_topk(logits_last)

    # policy metrics
    kept: Dict[int, List[float]] = {M: [] for M in M_GUARD_LIST}
    cosv: Dict[int, List[float]] = {M: [] for M in M_GUARD_LIST}
    relv: Dict[int, List[float]] = {M: [] for M in M_GUARD_LIST}

    for t in range(MAX_NEW_TOKENS):
        x = torch.tensor([[int(next_tok)]], device=device, dtype=torch.long)
        out = model(x, use_cache=True, past_key_values=past, output_attentions=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]

        A = out.attentions[PROBE_LAYER]      # [1,heads,1,k]
        attn = A[0, :, -1, :]                # [heads,k]
        k_len = int(attn.shape[-1])

        abs_pos_count = len(prompt_ids) + t + 1
        key_start = max(0, abs_pos_count - k_len)

        m, _ = wedge_masses_past_only(attn, labeler.chi_hist, key_start, abs_pos_count)
        keep_wedges = choose_wedges(m)

        # V cache
        _, V = get_layer_kv_from_cache(out.past_key_values, PROBE_LAYER)
        V = V[:, :, -k_len:, :]

        for M in M_GUARD_LIST:
            mask = build_mask_top_wedges_plus_infoset(
                attn_heads_k=attn,
                chi_hist=labeler.chi_hist,
                key_start=key_start,
                abs_pos_count=abs_pos_count,
                keep_wedges=keep_wedges,
                M_guard=M,
            )
            kept[M].append(float(mask.sum().item()) / max(float(k_len), 1.0))
            c, r = oracle_context_error(attn, V, mask)
            cosv[M].append(c)
            relv[M].append(r)

        # step kernel + sample
        labeler.consume_token(int(next_tok))
        next_tok = sample_topk(logits)

        if (t + 1) % 20 == 0:
            msg = f"[{t+1:3}]"
            for M in M_GUARD_LIST:
                msg += f" M{M}: kept={np.mean(kept[M][-20:]):.3f} rel={np.mean(relv[M][-20:]):.3f}"
            print(msg)

        if tok.eos_token_id is not None and next_tok == int(tok.eos_token_id):
            break

    print("\nSUMMARY")
    for M in M_GUARD_LIST:
        print(
            f"  wedges+M{M:2d}: kept={float(np.mean(kept[M])):.3f} "
            f"cos={float(np.mean(cosv[M])):.4f} rel={float(np.mean(relv[M])):.4f}"
        )
    print("  NOTE: still oracle-only (no speed change expected).")

    # after generation, print chi_hist distribution
    dist = np.bincount(np.array(labeler.chi_hist) & 3, minlength=4)
    print("chi_hist_dist:", dist.tolist(), "frac:", (dist / dist.sum()).round(3).tolist())
    print("keep_wedges_size (avg):", np.mean([len(choose_wedges(wedge_masses_past_only(attn, labeler.chi_hist, max(0, len(prompt_ids) + t - k_len), len(prompt_ids) + t + 1)[0])) for t in range(MAX_NEW_TOKENS)]))


def main() -> None:
    if not MODEL_DIR.exists() or not ATLAS_DIR.exists():
        print("Model or atlas not found.")
        return

    set_seed(SEED)
    print("Loading model (eager attention)...")
    model, tok = load_model_eager()

    for prompt in PROMPTS:
        print("\n" + "=" * 90)
        print("PROMPT:", prompt)
        set_seed(SEED)
        run_prompt(model, tok, prompt)


if __name__ == "__main__":
    main()