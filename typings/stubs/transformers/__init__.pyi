from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    overload,
)

import torch

Tensor = torch.Tensor


class ModelOutput(dict[str, Any]):
    """
    Minimal HF-like output:
    - attribute access (logits, hidden_states, ...)
    - dict-like indexing
    """
    logits: Tensor | None
    hidden_states: tuple[Tensor, ...] | None
    attentions: tuple[Tensor, ...] | None
    past_key_values: Any | None

    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...


class BatchEncoding(dict[str, Any]):
    """
    What tokenizers typically return.
    Common keys: input_ids, attention_mask.
    """
    input_ids: Tensor
    attention_mask: Tensor

    def __getattr__(self, name: str) -> Any: ...
    def to(self, device: str | torch.device) -> BatchEncoding: ...


class EmbeddingLayer:
    weight: Tensor


class InnerModel:
    embed_tokens: EmbeddingLayer
    layers: Any


class PreTrainedTokenizerBase:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> PreTrainedTokenizerBase: ...

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> list[int]: ...
    def decode(
        self,
        token_ids: int | Sequence[int],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: Any,
    ) -> BatchEncoding: ...
    @overload
    def __call__(
        self,
        text: Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: Any,
    ) -> BatchEncoding: ...
    @overload
    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any: ...

    @property
    def vocab_size(self) -> int: ...
    @property
    def eos_token_id(self) -> int | None: ...
    @property
    def pad_token_id(self) -> int | None: ...


class AutoTokenizer(PreTrainedTokenizerBase):
    ...


class PreTrainedModel(torch.nn.Module):
    config: Any

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> PreTrainedModel: ...


class AutoModelForCausalLM(PreTrainedModel):
    model: InnerModel
    lm_head: Any
    device: torch.device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        torch_dtype: torch.dtype | str | None = None,
        dtype: torch.dtype | str | None = None,
        device_map: str | Mapping[str, Any] | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> AutoModelForCausalLM: ...

    def __call__(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> ModelOutput: ...

    def generate(
        self,
        inputs: Tensor | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> Tensor: ...

    def to(self, *args: Any, **kwargs: Any) -> AutoModelForCausalLM: ...
    def eval(self) -> AutoModelForCausalLM: ...
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    @property
    def dtype(self) -> torch.dtype: ...
