"""
Byte tokenizer that computes GENE_Mic decomposition.

Input: raw text (UTF-8 string)
Output: dict with input_ids (bytes), families, micro_refs, introns

This is where BL1 begins. The tokenizer IS the first Byte Layer's
input preparation. No BPE. No learned vocabulary. Pure physics.

Can optionally wrap BolmoTokenizer for compatibility with existing
models during the conversion phase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast

from transformers import PreTrainedTokenizer

from . import physics


class PhysicsTokenizer(PreTrainedTokenizer):
    vocab_size = 256

    def __init__(self, **kwargs: Any):
        pad_token = kwargs.pop("pad_token", chr(0))
        bos_token = kwargs.pop("bos_token", None)
        eos_token = kwargs.pop("eos_token", None)
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    def encode_with_physics(self, text: str) -> Dict[str, List[int]]:
        """
        Returns input_ids + all GENE_Mic priors per byte.
        """
        raw = text.encode("utf-8", errors="replace")
        input_ids = list(raw)
        introns = [physics.intron(b) for b in input_ids]
        families = [(i >> 6) & 0x3 for i in introns]
        micro_refs = [i & 0x3F for i in introns]
        return {
            "input_ids": input_ids,
            "introns": introns,
            "families": families,
            "micro_refs": micro_refs,
        }

    def get_vocab(self) -> Dict[str, int]:
        vocab = {chr(i): i for i in range(256)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        return list(text)  # character-level for compatibility

    def _convert_token_to_id(self, token: str) -> int:
        if len(token) == 1:
            return ord(token)
        return 0

    def _convert_id_to_token(self, index: int) -> str:
        return chr(index & 0xFF)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            return token_ids_0 + token_ids_1
        return token_ids_0

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        return [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        return [0] * len(token_ids_0)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple:
        return ()

    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs: Any,
    ) -> str:
        flat: List[int] = cast(List[int], ids[0] if (ids and isinstance(ids[0], list)) else ids)
        try:
            return bytes(flat).decode("utf-8", errors="replace")
        except (TypeError, ValueError):
            return ""

    def from_bolmo_ids(self, bolmo_ids: List[int]) -> Dict[str, List[int]]:
        """
        Convert Bolmo token IDs to physics-encoded bytes.
        Bolmo uses BPE; this maps token IDs back to raw bytes where possible.
        """
        # Simplified: assume bolmo_ids are already byte-like or need conversion
        # In real use, would need Bolmo tokenizer decode + encode path
        input_ids = [b & 0xFF for b in bolmo_ids]
        return self.encode_with_physics(bytes(input_ids).decode("utf-8", errors="replace"))
