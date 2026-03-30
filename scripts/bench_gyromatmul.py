#!/usr/bin/env python3
"""GyroMatMul connection-level profiler for Bolmo generation."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from shutil import get_terminal_size
from typing import Any, Callable, DefaultDict, cast

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tools.gyrolabe.gyromatmul.convert import convert_bolmo
from src.tools.gyrolabe.gyromatmul.modules import GyroMatMulAttention, GyroMatMulMLP
from src.tools.gyrolabe.gyromatmul._build import get_ext
from src.tools.gyrolabe.bridges.bolmo_config import DEFAULT_BOLMO_MODEL_PATH, load_base_bolmo
from tests.tools.conftest import bolmo_tokenizer_from_model


class _ProfilerState:
    def __init__(self, attention_labels: dict[int, str], mlp_labels: dict[int, str], ext: Any) -> None:
        self.ext = ext
        self.attention_labels = attention_labels
        self.mlp_labels = mlp_labels
        self._active_layers: list[str] = []
        self._original_ext: dict[str, Callable[..., Any]] = {}
        self._original_attention_forward: Callable[..., Any] | None = None
        self._original_mlp_forward: Callable[..., Any] | None = None
        self._kernel_totals: DefaultDict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "total_ns": 0.0, "items": 0.0},
        )
        self._layer_totals: DefaultDict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "total_ns": 0.0},
        )
        self._layer_kernel_totals: DefaultDict[str, DefaultDict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"calls": 0.0, "total_ns": 0.0, "items": 0.0}),
        )

    def _active_layer(self) -> str:
        return self._active_layers[-1] if self._active_layers else "global"

    def _active_scope(self, label: str) -> Any:
        class _Scope:
            def __init__(self, parent: _ProfilerState, name: str) -> None:
                self.parent = parent
                self.name = name

            def __enter__(self) -> None:
                self.parent._active_layers.append(self.name)

            def __exit__(self, *_: Any) -> bool:
                if self.parent._active_layers:
                    self.parent._active_layers.pop()
                return False

        return _Scope(self, label)

    def _infer_path(self, kernel_name: str, args: tuple[Any, ...]) -> str:
        if not args:
            return kernel_name
        if kernel_name == "linear_forward_compiled":
            return f"{kernel_name} [compiled linear]"
        if kernel_name == "attention_decode_runtime_forward":
            return f"{kernel_name} [native decode runtime]"
        if kernel_name == "attention_decode_runtime_append_kv":
            return f"{kernel_name} [native decode cache append]"
        if kernel_name.startswith(("bmm_qk", "bmm_av")) and isinstance(args[0], torch.Tensor):
            if args[0].ndim == 4 and args[0].shape[2] == 1:
                return f"{kernel_name} [fluid bmm] [decode]"
            return f"{kernel_name} [fluid bmm] [prefill]"
        if kernel_name == "mlstm_step":
            return f"{kernel_name} [xlstm step]"
        return kernel_name

    def _estimate_items(self, kernel_name: str, args: tuple[Any, ...], out: Any) -> float:
        if kernel_name in {"linear_forward_compiled"}:
            return float(out.numel()) if isinstance(out, torch.Tensor) else 0.0
        if kernel_name == "attention_decode_runtime_forward":
            if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                return float(out[0].numel())
            if isinstance(args[0], torch.Tensor):
                return float(args[0].numel())
            return 0.0
        if kernel_name.startswith("bmm_qk") and len(args) >= 2:
            q, k = args[0], args[1]
            if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and q.ndim == 4 and k.ndim == 4:
                return float(q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2])
        if kernel_name == "attention_decode_runtime_append_kv" and len(args) >= 2:
            key = args[1]
            if isinstance(key, torch.Tensor):
                return float(key.numel())
        if kernel_name.startswith("bmm_av") and len(args) >= 2:
            a, v = args[0], args[1]
            if isinstance(a, torch.Tensor) and isinstance(v, torch.Tensor) and a.ndim == 4 and v.ndim == 4:
                return float(a.shape[0] * a.shape[1] * a.shape[2] * v.shape[3])
        if kernel_name == "mlstm_step" and len(args) >= 3:
            q, _, v = args[0], args[1], args[2]
            if isinstance(q, torch.Tensor) and isinstance(v, torch.Tensor) and q.ndim == 4 and v.ndim == 4:
                return float(q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3] * v.shape[3])
        return float(out.numel()) if isinstance(out, torch.Tensor) else 0.0

    def _record_kernel(self, kernel_name: str, args: tuple[Any, ...], out: Any, elapsed_ns: int) -> None:
        key = self._infer_path(kernel_name, args)
        layer = self._active_layer()
        items = self._estimate_items(kernel_name, args, out)
        t = self._kernel_totals[key]
        t["calls"] += 1.0
        t["total_ns"] += float(elapsed_ns)
        t["items"] += items
        lt = self._layer_kernel_totals[layer][key]
        lt["calls"] += 1.0
        lt["total_ns"] += float(elapsed_ns)
        lt["items"] += items

    def _make_ext_wrapper(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.perf_counter_ns()
            out = original(*args, **kwargs)
            self._record_kernel(name, args, out, time.perf_counter_ns() - start_ns)
            return out
        return wrapped

    def install(self, trace_layer_forward: bool = True) -> None:
        for name in (
            "linear_forward_compiled",
            "bmm_qk", "bmm_qk_gqa",
            "bmm_av", "bmm_av_gqa",
            "attention_decode_runtime_forward", "attention_decode_runtime_append_kv",
            "mlstm_step",
        ):
            original = getattr(self.ext, name, None)
            if original is None:
                continue
            self._original_ext[name] = cast(Callable[..., Any], original)
            setattr(self.ext, name, self._make_ext_wrapper(name, cast(Callable[..., Any], original)))

        self._original_attention_forward = GyroMatMulAttention.forward
        self._original_mlp_forward = GyroMatMulMLP.forward

        if trace_layer_forward:
            attn_labels = self.attention_labels
            mlp_labels = self.mlp_labels
            profiler = self

            def attention_forward(self_: Any, *args: Any, **kwargs: Any) -> Any:
                start_ns = time.perf_counter_ns()
                label = attn_labels.get(id(self_), f"attn:{id(self_)}")
                with profiler._active_scope(label):
                    out = cast(Any, profiler._original_attention_forward)(self_, *args, **kwargs)
                lt = profiler._layer_totals[label]
                lt["calls"] += 1.0
                lt["total_ns"] += float(time.perf_counter_ns() - start_ns)
                return out

            def mlp_forward(self_: Any, *args: Any, **kwargs: Any) -> Any:
                start_ns = time.perf_counter_ns()
                label = mlp_labels.get(id(self_), f"mlp:{id(self_)}")
                with profiler._active_scope(label):
                    out = cast(Any, profiler._original_mlp_forward)(self_, *args, **kwargs)
                lt = profiler._layer_totals[label]
                lt["calls"] += 1.0
                lt["total_ns"] += float(time.perf_counter_ns() - start_ns)
                return out

            GyroMatMulAttention.forward = cast(Any, attention_forward)
            GyroMatMulMLP.forward = cast(Any, mlp_forward)

    def uninstall(self) -> None:
        for name, original in self._original_ext.items():
            setattr(self.ext, name, original)
        if self._original_attention_forward is not None:
            GyroMatMulAttention.forward = cast(Any, self._original_attention_forward)
        if self._original_mlp_forward is not None:
            GyroMatMulMLP.forward = cast(Any, self._original_mlp_forward)
        self._original_ext.clear()
        self._original_attention_forward = None
        self._original_mlp_forward = None

    def kernel_totals(self) -> dict[str, dict[str, float]]:
        return dict(self._kernel_totals)

    def layer_totals(self) -> dict[str, dict[str, float]]:
        return dict(self._layer_totals)

    def layer_kernel_totals(self) -> dict[str, dict[str, dict[str, float]]]:
        return {layer: dict(kernel) for layer, kernel in self._layer_kernel_totals.items()}


def _collect_layer_labels(model: Any) -> tuple[dict[int, str], dict[int, str]]:
    attn: dict[int, str] = {}
    mlp: dict[int, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, GyroMatMulAttention):
            attn[id(module)] = name or "root"
        elif isinstance(module, GyroMatMulMLP):
            mlp[id(module)] = name or "root"
    return attn, mlp


def _run_generation(
    model: Any, input_ids: torch.Tensor, *, repeats: int, max_new_tokens: int,
) -> tuple[float, int]:
    def _once() -> torch.Tensor:
        with torch.no_grad():
            return cast(Any, model).generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    total = 0.0
    out: torch.Tensor | None = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        out = _once()
        total += time.perf_counter() - start
    assert out is not None
    return total / max(1, repeats), int(out.shape[-1] - input_ids.shape[-1])


def _run_prefill_only(model: Any, input_ids: torch.Tensor, *, repeats: int) -> float:
    def _once() -> None:
        with torch.no_grad():
            _ = cast(Any, model)(input_ids=input_ids)

    total = 0.0
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        _once()
        total += time.perf_counter() - start
    return total / max(1, repeats)


def _print_profile(
    total_ms: float,
    kernel_totals: dict[str, dict[str, float]],
    layer_totals: dict[str, dict[str, float]],
    layer_kernel_totals: dict[str, dict[str, dict[str, float]]],
) -> None:
    terminal_width = get_terminal_size((120, 20)).columns
    show_rate = terminal_width >= 88
    compact_kernel_name = max(16, min(36, terminal_width - 60))
    compact_layer_name = max(20, min(40, terminal_width - 56))
    compact_child_name = max(14, min(32, terminal_width - 64))

    def _short(name: str, max_len: int) -> str:
        if len(name) <= max_len:
            return name
        return f"{name[: max_len - 3]}..."

    def _shorten_layer(name: str) -> str:
        return (
            name.replace("model.local_encoder.layers.", "LEnc.")
            .replace("model.local_decoder.layers.", "LDec.")
            .replace("model.layers.", "L.")
        )

    total_ns = total_ms * 1_000_000.0
    print(f"  total: {total_ms:.3f} ms")

    if not kernel_totals:
        print("  no kernel calls recorded")
        return

    print("  kernels:")
    for name, d in sorted(kernel_totals.items(), key=lambda kv: kv[1]["total_ns"], reverse=True):
        ms = d["total_ns"] / 1e6
        pct = d["total_ns"] / total_ns * 100 if total_ns > 0 else 0
        calls = int(d["calls"])
        avg = ms / max(1, calls)
        rate = d["items"] / (d["total_ns"] / 1e9) if d["total_ns"] > 0 else 0
        short_name = _short(name, compact_kernel_name)
        if show_rate:
            print(
                f"    K: {short_name} | c={calls} | "
                f"t={ms:.3f}ms | {pct:.1f}% | a={avg:.3f}ms | r={rate:.0f}/s"
            )
        else:
            print(
                f"    K: {short_name} | c={calls} | "
                f"t={ms:.3f}ms | {pct:.1f}% | a={avg:.3f}ms"
            )

    print("  layers:")
    for layer, d in sorted(layer_totals.items(), key=lambda kv: kv[1]["total_ns"], reverse=True):
        ms = d["total_ns"] / 1e6
        pct = d["total_ns"] / total_ns * 100 if total_ns > 0 else 0
        short_layer = _short(_shorten_layer(layer), compact_layer_name)
        print(f"    L: {short_layer} | c={int(d['calls'])} | t={ms:.3f}ms | {pct:.1f}%")
        kernels = layer_kernel_totals.get(layer, {})
        kernel_children = sorted(kernels.items(), key=lambda kv: kv[1]["total_ns"], reverse=True)
        for kn, kd in kernel_children:
            kms = kd["total_ns"] / 1e6
            kpct = kd["total_ns"] / max(1.0, d["total_ns"]) * 100
            short_kernel = _short(kn, compact_child_name)
            print(f"      K: {short_kernel} | c={int(kd['calls'])} | t={kms:.3f}ms | {kpct:.1f}%")


def _build_model(model_path: str, n_bits: int) -> Any:
    raw = cast(Any, load_base_bolmo(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True))
    converted = convert_bolmo(raw, n_bits=n_bits)
    converted.eval()
    return converted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile GyroMatMul generation")
    p.add_argument("--model-path", default=str(DEFAULT_BOLMO_MODEL_PATH))
    p.add_argument("--prompt", default="The weather today is")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--n-bits", type=int, default=12)
    p.add_argument("--prefill-only", action="store_true")
    p.add_argument("--skip-gc", action="store_true")
    return p.parse_args()


def _run_warmup_generate(model: Any, input_ids: torch.Tensor, *, max_new_tokens: int, warmup: int) -> None:
    def _once() -> torch.Tensor:
        with torch.no_grad():
            return cast(Any, model).generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    for _ in range(max(0, warmup)):
        _once()


def _run_warmup_prefill(model: Any, input_ids: torch.Tensor, *, warmup: int) -> None:
    def _once() -> None:
        with torch.no_grad():
            _ = cast(Any, model)(input_ids=input_ids)

    for _ in range(max(0, warmup)):
        _once()


def _normalize_totals_for_average(kernel_totals, layer_totals, layer_kernel_totals, repeats):
    reps = max(1, float(repeats))
    return (
        {name: {"calls": value["calls"] / reps, "total_ns": value["total_ns"] / reps, "items": value["items"] / reps}
         for name, value in kernel_totals.items()},
        {name: {"calls": value["calls"] / reps, "total_ns": value["total_ns"] / reps}
         for name, value in layer_totals.items()},
        {
            layer: {
                name: {"calls": value["calls"] / reps, "total_ns": value["total_ns"] / reps, "items": value["items"] / reps}
                for name, value in kernel_map.items()
            }
            for layer, kernel_map in layer_kernel_totals.items()
        },
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)

    print("GyroMatMul profiler")
    print(f"  model={args.model_path}")
    print(f"  n_bits={args.n_bits}")
    model = _build_model(args.model_path, args.n_bits)
    hook_status: dict[str, bool] = {}
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls == "GyroMatMulAttention":
            hook_status[f"{name}.native_attn_weights"] = bool(
                getattr(module, "_use_native_attention_weights", False)
            )

    active = sum(1 for v in hook_status.values() if v)
    total = len(hook_status)
    if total:
        print(f"  feature flags: {active}/{total} active")
    else:
        print("  feature flags: none configured")
    if active < total:
        for k, v in hook_status.items():
            if not v:
                print(f"    MISSING: {k}")

    tokenizer = bolmo_tokenizer_from_model(model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    ext = get_ext()
    if ext is None:
        raise RuntimeError("GyroMatMul extension not available")

    attn_labels, mlp_labels = _collect_layer_labels(model)
    profiler = _ProfilerState(attn_labels, mlp_labels, ext)
    ext_has_linear_regime = (
        hasattr(ext, "reset_linear_regime_counters")
        and hasattr(ext, "get_linear_regime_counters")
    )
    if ext_has_linear_regime:
        try:
            ext.reset_linear_regime_counters()
        except Exception:
            ext_has_linear_regime = False

    generated_tokens = 0
    try:
        if args.prefill_only:
            _run_warmup_prefill(model, input_ids, warmup=args.warmup)
            profiler.install()
            elapsed = _run_prefill_only(model, input_ids, repeats=args.repeats)
        else:
            _run_warmup_generate(model, input_ids, max_new_tokens=args.max_new_tokens, warmup=args.warmup)
            profiler.install()
            elapsed, generated_tokens = _run_generation(
                model, input_ids, repeats=args.repeats, max_new_tokens=args.max_new_tokens
            )
    finally:
        if ext_has_linear_regime:
            try:
                dense, bulk8, matmul = ext.get_linear_regime_counters()
            except Exception:
                pass
            else:
                total = dense + bulk8 + matmul
                if total > 0:
                    print(
                        f"  linear regimes: dense={dense} ({dense / total * 100.0:0.1f}%), "
                        f"bulk8={bulk8} ({bulk8 / total * 100.0:0.1f}%), "
                        f"matmul={matmul} ({matmul / total * 100.0:0.1f}%)"
                    )
                else:
                    print("  linear regimes: dense=0, bulk8=0, matmul=0")
        profiler.uninstall()

    if not args.skip_gc:
        gc.collect()

    mode = "prefill" if args.prefill_only else "generate"
    print(f"  mode={mode}  prompt_tokens={input_ids.shape[-1]}", end="")
    if not args.prefill_only:
        print(f"  max_new={args.max_new_tokens}  generated={generated_tokens}", end="")
    print()
    kernel_totals, layer_totals, layer_kernel_totals = _normalize_totals_for_average(
        profiler.kernel_totals(),
        profiler.layer_totals(),
        profiler.layer_kernel_totals(),
        repeats=args.repeats,
    )
    _print_profile(elapsed * 1000, kernel_totals, layer_totals, layer_kernel_totals)


if __name__ == "__main__":
    main()
