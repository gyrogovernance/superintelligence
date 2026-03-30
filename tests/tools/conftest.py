from __future__ import annotations

import os
import time
import warnings
from typing import Any, cast

import numpy as np
import pytest
import torch


def _configure_offline_bolmo_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def pytest_configure(config) -> None:
    _configure_offline_bolmo_env()
    for pattern, category in [
        (".*rope_config_validation.*", FutureWarning),
        (".*SwigPyPacked has no __module__ attribute.*", DeprecationWarning),
        (".*SwigPyObject has no __module__ attribute.*", DeprecationWarning),
        (".*swigvarlink has no __module__ attribute.*", DeprecationWarning),
    ]:
        config.addinivalue_line("filterwarnings", f"ignore:{pattern}:{category.__name__}")
        warnings.filterwarnings("ignore", message=pattern, category=category)


# ---------------------------------------------------------------------------
# Shared helpers (used by tests and benchmarks)
# ---------------------------------------------------------------------------


def bolmo_tokenizer_from_model(model: Any) -> Any:
    tokenizer = getattr(model, "model", None) and getattr(model.model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AssertionError("Bolmo model did not provide tokenizer")
    sample_ids = tokenizer(" hello", return_tensors="pt").input_ids
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and sample_ids.max() >= int(vocab_size):
        raise AssertionError(
            f"Tokenizer produced id {int(sample_ids.max())}, but vocab_size is {vocab_size}"
        )
    return tokenizer


def measure_generation_ms(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    repeats: int = 3,
    warmup: int = 1,
) -> tuple[float, int]:
    for _ in range(max(0, warmup)):
        with torch.no_grad():
            _ = cast(Any, model).generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    times: list[float] = []
    out = None
    for _ in range(max(1, repeats)):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = cast(Any, model).generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            times.append(time.perf_counter() - t0)
    assert out is not None
    times.sort()
    tokens_added = int(out.shape[-1] - input_ids.shape[-1])
    return times[len(times) // 2], tokens_added


def random_i32_matrix(
    rows: int, cols: int, seed: int, *, low: int = -1000, high: int = 1001,
) -> np.ndarray:
    return np.random.RandomState(seed).randint(low, high, size=(rows, cols)).astype(np.int32)


def assert_array_exact(name: str, result: Any, expected: Any) -> None:
    r = result.numpy() if isinstance(result, torch.Tensor) else np.asarray(result)
    e = expected.numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
    if not np.array_equal(r, e):
        diff = int(np.max(np.abs(r.astype(np.int64) - e.astype(np.int64))))
        pytest.fail(f"{name}: max diff = {diff}")


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    denom = torch.maximum(
        expected.abs(),
        torch.full_like(expected.abs(), float(expected.abs().mean().detach()) * 0.2),
    )
    return (actual - expected).abs() / denom


def find_bolmo_embedding(raw_bolmo: Any) -> torch.nn.Embedding | None:
    candidates: list[tuple[tuple[str, ...], bool]] = [
        (("model", "local_encoder", "byte_embedding"), False),
        (("model", "local_encoder", "subword_embedding"), False),
        (("local_encoder", "byte_embedding"), False),
        (("local_encoder", "subword_embedding"), False),
        (("model", "get_input_embeddings"), True),
        (("get_input_embeddings",), True),
    ]
    for path, should_call in candidates:
        target: Any = raw_bolmo
        try:
            for key in path:
                target = getattr(target, key)
            if should_call:
                target = target()
            if isinstance(target, torch.nn.Embedding):
                return target
        except (AttributeError, TypeError):
            continue
    return None