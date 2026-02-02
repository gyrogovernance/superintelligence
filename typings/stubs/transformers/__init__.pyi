from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor


class ModelOutput:
    logits: Optional[Tensor]
    hidden_states: Optional[Tuple[Tensor, ...]]
    attentions: Optional[Tuple[Tensor, ...]]
    past_key_values: Optional[Any]
    loss: Optional[Tensor]
    last_hidden_state: Optional[Tensor]

    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...
    def keys(self) -> Any: ...
    def values(self) -> Any: ...
    def items(self) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def to_tuple(self) -> Tuple[Any, ...]: ...


class BatchEncoding:
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Optional[Tensor]

    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> Any: ...
    def values(self) -> Any: ...
    def items(self) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def to(self, device: Any) -> "BatchEncoding": ...


class PretrainedConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    model_type: str

    def __init__(self, **kwargs: Any) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json_string(self) -> str: ...
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None: ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> "PretrainedConfig": ...

    def __getattr__(self, name: str) -> Any: ...


class PreTrainedTokenizerBase:
    vocab_size: int
    model_max_length: int
    padding_side: str
    truncation_side: str
    eos_token: Optional[str]
    eos_token_id: Optional[int]
    bos_token: Optional[str]
    bos_token_id: Optional[int]
    pad_token: Optional[str]
    pad_token_id: Optional[int]
    unk_token: Optional[str]
    unk_token_id: Optional[int]
    sep_token: Optional[str]
    sep_token_id: Optional[int]
    cls_token: Optional[str]
    cls_token_id: Optional[int]
    mask_token: Optional[str]
    mask_token_id: Optional[int]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *model_args: Any, trust_remote_code: bool = False, **kwargs: Any) -> "PreTrainedTokenizerBase": ...

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> list[int]: ...
    def decode(self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any) -> str: ...
    def batch_decode(self, sequences: Any, skip_special_tokens: bool = False, **kwargs: Any) -> list[str]: ...

    def __call__(
        self,
        text: Any = None,
        text_pair: Any = None,
        text_target: Any = None,
        text_pair_target: Any = None,
        add_special_tokens: bool = True,
        padding: Any = False,
        truncation: Any = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> BatchEncoding: ...

    def convert_tokens_to_ids(self, tokens: Any) -> Any: ...
    def convert_ids_to_tokens(self, ids: Any, skip_special_tokens: bool = False) -> Any: ...
    def tokenize(self, text: str, **kwargs: Any) -> list[str]: ...
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> Tuple[str, ...]: ...

    def __len__(self) -> int: ...
    def __getattr__(self, name: str) -> Any: ...


class AutoTokenizer(PreTrainedTokenizerBase):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *model_args: Any, trust_remote_code: bool = False, **kwargs: Any) -> "AutoTokenizer": ...


class PreTrainedModel(torch.nn.Module):
    config: PretrainedConfig
    device: torch.device
    dtype: torch.dtype
    generation_config: Any
    name_or_path: str

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Any,
        *model_args: Any,
        config: Optional[PretrainedConfig] = None,
        cache_dir: Optional[str] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Any] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> "PreTrainedModel": ...

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict: Optional[dict[str, Any]] = None,
        save_function: Any = None,
        push_to_hub: bool = False,
        max_shard_size: str = "5GB",
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding: ...
    def get_input_embeddings(self) -> torch.nn.Module: ...
    def set_input_embeddings(self, value: torch.nn.Module) -> None: ...
    def get_output_embeddings(self) -> Optional[torch.nn.Module]: ...
    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None: ...
    def tie_weights(self) -> None: ...
    def gradient_checkpointing_enable(self) -> None: ...
    def gradient_checkpointing_disable(self) -> None: ...

    def __call__(self, *args: Any, **kwargs: Any) -> ModelOutput: ...
    def forward(self, *args: Any, **kwargs: Any) -> ModelOutput: ...

    def __getattr__(self, name: str) -> Any: ...


class GenerationConfig:
    max_length: int
    max_new_tokens: Optional[int]
    min_length: int
    do_sample: bool
    num_beams: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    bos_token_id: Optional[int]

    def __init__(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


class GenerationMixin:
    def generate(
        self,
        inputs: Optional[Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[Any] = None,
        stopping_criteria: Optional[Any] = None,
        prefix_allowed_tokens_fn: Optional[Any] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional[Any] = None,
        streamer: Optional[Any] = None,
        negative_prompt_ids: Optional[Tensor] = None,
        negative_prompt_attention_mask: Optional[Tensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Tensor: ...


class AutoModelForCausalLM(PreTrainedModel, GenerationMixin):
    model: Any
    lm_head: Any

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Any,
        *model_args: Any,
        config: Optional[PretrainedConfig] = None,
        cache_dir: Optional[str] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Any] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> "AutoModelForCausalLM": ...

    def __call__(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> ModelOutput: ...


class AutoModel(PreTrainedModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Any,
        *model_args: Any,
        config: Optional[PretrainedConfig] = None,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "AutoModel": ...


class AutoModelForSequenceClassification(PreTrainedModel):
    classifier: Any
    num_labels: int

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Any,
        *model_args: Any,
        config: Optional[PretrainedConfig] = None,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "AutoModelForSequenceClassification": ...


class AutoModelForTokenClassification(PreTrainedModel):
    classifier: Any
    num_labels: int

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Any,
        *model_args: Any,
        config: Optional[PretrainedConfig] = None,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "AutoModelForTokenClassification": ...


class AutoConfig:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> PretrainedConfig: ...


class Trainer:
    model: PreTrainedModel
    args: Any
    train_dataset: Any
    eval_dataset: Any
    tokenizer: Optional[PreTrainedTokenizerBase]

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        args: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        optimizers: Tuple[Optional[Any], Optional[Any]] = (None, None),
        preprocess_logits_for_metrics: Optional[Any] = None,
    ) -> None: ...

    def train(self, resume_from_checkpoint: Optional[Any] = None, **kwargs: Any) -> Any: ...
    def evaluate(self, eval_dataset: Optional[Any] = None, **kwargs: Any) -> dict[str, float]: ...
    def predict(self, test_dataset: Any, **kwargs: Any) -> Any: ...
    def save_model(self, output_dir: Optional[str] = None, **kwargs: Any) -> None: ...
    def save_state(self) -> None: ...
    def log(self, logs: dict[str, float]) -> None: ...


class TrainingArguments:
    output_dir: str
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: Optional[int]
    warmup_steps: int
    warmup_ratio: float
    gradient_accumulation_steps: int
    fp16: bool
    bf16: bool
    evaluation_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    report_to: Any
    remove_unused_columns: bool
    label_names: Optional[list[str]]

    def __init__(
        self,
        output_dir: str,
        overwrite_output_dir: bool = False,
        do_train: bool = False,
        do_eval: bool = False,
        do_predict: bool = False,
        evaluation_strategy: str = "no",
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        num_train_epochs: float = 3.0,
        max_steps: int = -1,
        lr_scheduler_type: str = "linear",
        warmup_ratio: float = 0.0,
        warmup_steps: int = 0,
        logging_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        seed: int = 42,
        fp16: bool = False,
        bf16: bool = False,
        dataloader_num_workers: int = 0,
        load_best_model_at_end: bool = False,
        report_to: Optional[Any] = None,
        **kwargs: Any,
    ) -> None: ...

    def __getattr__(self, name: str) -> Any: ...


class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Any
    max_length: Optional[int]
    pad_to_multiple_of: Optional[int]
    return_tensors: str

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Any = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> None: ...

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]: ...


class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool
    mlm_probability: float
    pad_to_multiple_of: Optional[int]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> None: ...

    def __call__(self, examples: list[Any]) -> dict[str, Tensor]: ...


def set_seed(seed: int) -> None: ...