"""
Shared setup and Bolmo loading for Bolmo Kernel Port lab.
Paths are resolved from this file so scripts run from blomo_port/ or repo root.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings(
    "ignore", message=".*rope_config_validation.*",
    category=FutureWarning, module="transformers.*",
)

# Repo root (superintelligence): blomo_port -> secret_lab_ignore -> repo root
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suffix automaton cache (optional global speedup for expand_byte_ids)
SUFFIX_AUTOMATON_PATH = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "suffix_automaton.npz"

import torch

from src.router.constants import mask12_for_byte, popcount, unpack_state, vertex_charge_from_mask


def _patch_rope_default_for_bolmo() -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute(config, device=None, seq_len=None, layer_type=None):
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

    ROPE_INIT_FUNCTIONS["default"] = _compute


def _patch_rope_initialization() -> None:
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
        raise RuntimeError(f"dolma2 tokenizer not cached at {local_dir}")
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


class MaskState:
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
            if self.all(): return x[[]]
            if not self.any(): return x
            return x[self.inv_mask]
        else:
            if self.all(): return x
            if not self.any(): return x[[]]
            return x[self.mask]

    def selective_put(self, x: torch.Tensor, out: torch.Tensor, inv: bool = False) -> None:
        if inv:
            if self.all(): return
            if not self.any(): out.copy_(x); return
            out[self.inv_mask] = x
        else:
            if self.all(): out.copy_(x); return
            if not self.any(): return
            out[self.mask] = x

    def selective_add(self, x: int, out: torch.Tensor, inv: bool = False) -> None:
        if inv:
            if self.all(): return
            if not self.any(): out.add_(x); return
            out[self.inv_mask] += x
        else:
            if self.all(): out.add_(x); return
            if not self.any(): return
            out[self.mask] += x


def token_to_byte_and_fused(token_id: int, token_offset: int) -> tuple[Optional[int], bool]:
    t = int(token_id)
    if t < token_offset:
        return (None, False)
    if token_offset <= t < token_offset + 256:
        return (t - token_offset, False)
    if token_offset + 256 <= t < token_offset + 512:
        return (t - (token_offset + 256), True)
    return (None, False)


def maybe_patch_expand_byte_ids(tokenizer: Any) -> None:
    """
    If suffix_automaton.npz exists, patch tokenizer.expand_byte_ids to use the
    automaton (same semantics, ~4x faster). All modules using expand_byte_ids
    then benefit transparently. Build the npz with module 8 or adaptors/suffix_adaptor.py.
    """
    if not SUFFIX_AUTOMATON_PATH.exists():
        return
    try:
        from adaptors.suffix_adaptor import SuffixAutomaton, expand_byte_ids_with_automaton
    except Exception:
        return
    automaton = SuffixAutomaton.load(SUFFIX_AUTOMATON_PATH)
    orig = tokenizer.expand_byte_ids

    def fast_expand(byte_ids: list[int], n_last: Optional[int] = None) -> list[int]:
        if n_last is not None:
            return orig(byte_ids, n_last=n_last)
        return expand_byte_ids_with_automaton(byte_ids, automaton, tokenizer)

    tokenizer.expand_byte_ids = fast_expand


def encode_no_specials(tokenizer: Any, text: str) -> list[int]:
    enc = tokenizer(text, return_tensors=None, add_special_tokens=False)
    if isinstance(enc, dict) and "input_ids" in enc:
        ids = enc["input_ids"]
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            return [int(x) for x in ids[0]]
        return [int(x) for x in ids]
    if hasattr(tokenizer, "encode"):
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    return [int(x) for x in enc]


def load_bolmo(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _patch_rope_default_for_bolmo()
    _patch_rope_initialization()
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

    dolma2_dir = ensure_dolma2_tokenizer(PROJECT_ROOT / "data" / "models" / "dolma2-tokenizer")
    tc = model.model.tokenizer_config
    tc.original_identifier = str(dolma2_dir)
    tokenizer = tc.build()
    tokenizer.config.eos_token_id = 2
    try:
        model.config.eos_token_id = 2
    except Exception:
        pass
    model.model._tokenizer = tokenizer

    print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
