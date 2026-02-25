"""
Decode streaming adaptor: hook-based streaming suffix for generate().

Bolmo's generate() supports an optional tokenizer._decode_expand_hook. When set,
the hook is called at three points:
  - "init": after rollback, with byte_input_ids (prefix). Hook inits internal state.
  - "get_expanded": each step, with generated, is_first_forward, etc. Returns expanded
    tensor or None to use default expand_byte_ids.
  - "append": after generated = torch.cat(..., next_tokens). Hook updates state.

This adaptor implements that hook using the suffix automaton so decode uses O(1)
suffix state instead of O(L) expand_byte_ids(..., n_last=1). All logic stays in
the lab; modeling_bolmo.py only has minimal hook call sites.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as script from any cwd: blomo_port must be on path for common
_adaptors_dir = Path(__file__).resolve().parent
_blomo_port = _adaptors_dir.parent
if str(_blomo_port) not in sys.path:
    sys.path.insert(0, str(_blomo_port))

from typing import Any, Optional

import torch

from common import PROJECT_ROOT


def _load_automaton(path: Optional[Path] = None) -> Any:
    if path is None:
        path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "suffix_automaton.npz"
    if not path.exists():
        raise FileNotFoundError(f"suffix_automaton.npz not found: {path}")
    from adaptors.suffix_adaptor import SuffixAutomaton
    return SuffixAutomaton.load(path)


def _special_expansion(tid: int, tok: Any, hf: Any, unk_id: int, pad_id: int) -> int:
    if getattr(tok, "bos_token_id", None) is not None and tid == tok.bos_token_id:
        hb = getattr(hf, "bos_token_id", None)
        return int(hb) if hb is not None else unk_id
    if getattr(tok, "pad_token_id", None) is not None and tid == tok.pad_token_id:
        hp = getattr(hf, "pad_token_id", None)
        return int(hp) if hp is not None else pad_id
    if getattr(tok, "eos_token_id", None) is not None and tid == tok.eos_token_id:
        he = getattr(hf, "eos_token_id", None)
        return int(he) if he is not None else unk_id
    return unk_id if getattr(hf, "unk_token_id", None) is not None else pad_id


class DecodeStreamingHook:
    """
    Stateful hook for tokenizer._decode_expand_hook. Implements streaming
    suffix state (ac_state, eid_last) so generate() never calls expand_byte_ids
    in the decode loop.
    """

    def __init__(self, automaton: Any, tokenizer: Any):
        self.automaton = automaton
        self.tokenizer = tokenizer
        self.offset = int(getattr(tokenizer, "offset", 4))
        hf = tokenizer.hf_tokenizer
        self.unk_id = int(getattr(hf, "unk_token_id", 0))
        pad = getattr(hf, "pad_token_id", None)
        self.pad_id = int(pad) if pad is not None else self.unk_id
        self.bpe_end_id = int(getattr(tokenizer, "bpe_token_end_id", 3))

        self.ac_state: list[int] = []
        self.eid_last: list[int] = []
        self.first_forward_expanded_list: list[list[int]] = []

    def __call__(self, phase: str, **kwargs: Any) -> Optional[torch.Tensor]:
        if phase == "init":
            self._init(**kwargs)
            return None
        if phase == "get_expanded":
            return self._get_expanded(**kwargs)
        if phase == "append":
            self._append(**kwargs)
            return None
        return None

    def _init(self, byte_input_ids: torch.Tensor, batch_size: int, tokenizer: Any, **kwargs: Any) -> None:
        tok = tokenizer
        ac_state = [0] * batch_size
        eid_last = [0] * batch_size
        first_forward_expanded_list: list[list[int]] = []

        for i in range(batch_size):
            full_ids = byte_input_ids[i, :].tolist()
            st = 0
            eids: list[int] = []
            for tid in full_ids:
                if tid == self.bpe_end_id:
                    eids.append(eids[-1] if eids else self.unk_id)
                    continue
                if tid < self.offset:
                    st = 0
                    eids.append(_special_expansion(tid, tok, tok.hf_tokenizer, self.unk_id, self.pad_id))
                    continue
                if tid >= self.offset + 256:
                    byte_val = tid - (self.offset + 256)
                else:
                    byte_val = tid - self.offset
                st = int(self.automaton.next_state[st, byte_val])
                eid = int(self.automaton.best_eid[st])
                eids.append(self.unk_id if eid == -1 else eid)
            ac_state[i] = st
            eid_last[i] = eids[-1] if eids else self.unk_id
            first_forward_expanded_list.append(eids)

        self.ac_state = ac_state
        self.eid_last = eid_last
        self.first_forward_expanded_list = first_forward_expanded_list

    def _get_expanded(
        self,
        generated: torch.Tensor,
        input_ids_for_model: torch.Tensor,
        is_first_forward: bool,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs: Any,
    ) -> torch.Tensor:
        expanded = torch.zeros_like(input_ids_for_model, device=device, dtype=dtype)
        if is_first_forward:
            for i in range(batch_size):
                expanded[i, :] = torch.tensor(
                    self.first_forward_expanded_list[i],
                    device=device,
                    dtype=dtype,
                )
        else:
            for i in range(batch_size):
                expanded[i, 0] = self.eid_last[i]
        return expanded

    def _append(self, next_tokens: torch.Tensor, batch_size: int, tokenizer: Any, **kwargs: Any) -> None:
        tok = tokenizer
        hf = tok.hf_tokenizer
        for i in range(batch_size):
            tid = next_tokens[i].item()
            if tid == self.bpe_end_id:
                continue
            if tid < self.offset:
                self.ac_state[i] = 0
                self.eid_last[i] = _special_expansion(tid, tok, hf, self.unk_id, self.pad_id)
            else:
                if tid >= self.offset + 256:
                    byte_val = tid - (self.offset + 256)
                else:
                    byte_val = tid - self.offset
                st = int(self.automaton.next_state[self.ac_state[i], byte_val])
                eid = int(self.automaton.best_eid[st])
                self.ac_state[i] = st
                self.eid_last[i] = self.unk_id if eid == -1 else eid


def install_decode_streaming_hook(tokenizer: Any, automaton_path: Optional[Path] = None) -> DecodeStreamingHook:
    """Set tokenizer._decode_expand_hook to a streaming suffix hook. Returns the hook instance."""
    automaton = _load_automaton(automaton_path)
    hook = DecodeStreamingHook(automaton, tokenizer)
    tokenizer._decode_expand_hook = hook  # type: ignore[attr-defined]
    return hook


def uninstall_decode_streaming_hook(tokenizer: Any) -> None:
    """Remove tokenizer._decode_expand_hook if present."""
    if hasattr(tokenizer, "_decode_expand_hook"):
        del tokenizer._decode_expand_hook  # type: ignore[attr-defined]


def main() -> None:
    """Load automaton, tokenizer, install hook; validates setup."""
    from common import load_bolmo
    print("Decode streaming adaptor")
    path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "suffix_automaton.npz"
    if not path.exists():
        print(f"Missing: {path}")
        print("Run suffix_adaptor.py first to build the automaton.")
        return
    automaton = _load_automaton(path)
    print(f"  Loaded automaton: {automaton.num_states} states")
    model_dir = PROJECT_ROOT / "data" / "models" / "Bolmo-1B"
    _, tokenizer = load_bolmo(model_dir, torch.device("cpu"))
    hook = install_decode_streaming_hook(tokenizer, path)
    print(f"  Hook installed (offset={hook.offset}, bpe_end_id={hook.bpe_end_id})")
    uninstall_decode_streaming_hook(tokenizer)
    print("  OK")


if __name__ == "__main__":
    main()
