from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union, overload, Literal

import torch

Tensor = torch.Tensor


class ModelOutput(dict[str, Any]):
    """
    Minimal HF-like output:
    - attribute access (logits, hidden_states, ...)
    - dict-like indexing
    """
    logits: Optional[Tensor]
    hidden_states: Optional[Tuple[Tensor, ...]]
    attentions: Optional[Tuple[Tensor, ...]]
    past_key_values: Optional[Any]

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
    def to(self, device: Union[str, torch.device]) -> BatchEncoding: ...


class EmbeddingLayer:
    weight: Tensor


class InnerModel:
    embed_tokens: EmbeddingLayer
    layers: Any


class PreTrainedTokenizerBase:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args: Any,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> PreTrainedTokenizerBase: ...

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> list[int]: ...
    def decode(
        self,
        token_ids: Union[int, Sequence[int]],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: Any,
    ) -> BatchEncoding: ...
    @overload
    def __call__(
        self,
        text: Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: Any,
    ) -> BatchEncoding: ...
    @overload
    def __call__(
        self,
        text: Union[str, Sequence[str]],
        *,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Any: ...

    @property
    def vocab_size(self) -> int: ...
    @property
    def eos_token_id(self) -> Optional[int]: ...
    @property
    def pad_token_id(self) -> Optional[int]: ...


class AutoTokenizer(PreTrainedTokenizerBase):
    ...


class PreTrainedModel(torch.nn.Module):
    config: Any

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
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
        pretrained_model_name_or_path: Union[str, Path],
        *model_args: Any,
        torch_dtype: Optional[Union[torch.dtype, str]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        device_map: Optional[Union[str, Mapping[str, Any]]] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> AutoModelForCausalLM: ...

    def __call__(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> ModelOutput: ...

    def generate(
        self,
        inputs: Optional[Tensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
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