"""
OLMo Forward Pass Trace

Hooks into the actual OLMo-3-7B model to trace weight access patterns
during a forward pass. This shows exactly what fires and in what order.

For Gyroscopic ASI: Real data on model execution patterns.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModuleTrace:
    """Trace of a single module's forward pass."""
    name: str
    module_type: str
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    weight_info: dict[str, Any] | None
    order: int


class OLMoTracer:
    """
    Trace all module calls during OLMo forward pass.
    """
    def __init__(self):
        self.traces: list[ModuleTrace] = []
        self._order = 0
        self._handles: list[Any] = []

    def _hook_fn(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
            # Input shapes
            input_shapes = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(list(inp.shape))

            # Output shapes
            output_shapes = []
            if isinstance(outputs, torch.Tensor):
                output_shapes.append(list(outputs.shape))
            elif isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor):
                        output_shapes.append(list(out.shape))

            # Weight info for linear layers
            weight_info = None
            weight = getattr(module, 'weight', None)
            if weight is not None:
                weight_info = {
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                    "numel": weight.numel(),
                }

                self.traces.append(ModuleTrace(
                    name=name,
                    module_type=type(module).__name__,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    weight_info=weight_info,
                    order=self._order,
                ))
            self._order += 1

        return hook

    def attach(self, model: nn.Module | Any) -> None:
        """Attach hooks to all submodules."""
        for name, module in model.named_modules():
            if name == "":
                continue
            # Skip container modules, only trace leaf modules
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(self._hook_fn(name))
                self._handles.append(handle)

    def detach(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summarize(self) -> dict[str, Any]:
        """Summarize traced operations."""
        by_type: dict[str, int] = {}
        total_weights = 0
        linear_ops = []

        for trace in self.traces:
            by_type[trace.module_type] = by_type.get(trace.module_type, 0) + 1
            if trace.weight_info:
                total_weights += trace.weight_info["numel"]
            if trace.module_type == "Linear":
                linear_ops.append({
                    "name": trace.name,
                    "weight_shape": trace.weight_info["shape"] if trace.weight_info else None,
                    "input": trace.input_shapes,
                    "output": trace.output_shapes,
                })

        return {
            "total_modules_called": len(self.traces),
            "by_type": by_type,
            "total_weight_params": total_weights,
            "linear_ops": linear_ops[:20],  # First 20 linear ops
        }


def trace_single_layer(model: Any, layer_idx: int) -> dict[str, Any]:
    """
    Detailed trace of a single transformer layer.
    Shows actual components from the model.
    """
    layer = model.model.layers[layer_idx]

    info = {
        "layer_idx": layer_idx,
        "components": {},
    }

    # Self Attention
    attn = layer.self_attn
    info["components"]["self_attn"] = {
        "type": type(attn).__name__,
    }

    # Attention projections
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if hasattr(attn, proj_name):
            proj = getattr(attn, proj_name)
            info["components"]["self_attn"][proj_name] = {
                "weight_shape": list(proj.weight.shape),
                "has_bias": proj.bias is not None,
            }

    # QK normalization (OLMo3 specific)
    for norm_name in ['q_norm', 'k_norm']:
        if hasattr(attn, norm_name):
            norm = getattr(attn, norm_name)
            info["components"]["self_attn"][norm_name] = {
                "type": type(norm).__name__,
                "weight_shape": list(norm.weight.shape) if hasattr(norm, 'weight') else None,
            }

    # Post-attention LayerNorm
    if hasattr(layer, 'post_attention_layernorm'):
        ln = layer.post_attention_layernorm
        info["components"]["post_attention_layernorm"] = {
            "type": type(ln).__name__,
            "weight_shape": list(ln.weight.shape) if hasattr(ln, 'weight') else None,
        }

    # MLP
    mlp = layer.mlp
    info["components"]["mlp"] = {
        "type": type(mlp).__name__,
    }

    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        if hasattr(mlp, proj_name):
            proj = getattr(mlp, proj_name)
            info["components"]["mlp"][proj_name] = {
                "weight_shape": list(proj.weight.shape),
                "has_bias": proj.bias is not None,
            }

    # Post-feedforward LayerNorm (OLMo3 specific)
    if hasattr(layer, 'post_feedforward_layernorm'):
        ln = layer.post_feedforward_layernorm
        info["components"]["post_feedforward_layernorm"] = {
            "type": type(ln).__name__,
            "weight_shape": list(ln.weight.shape) if hasattr(ln, 'weight') else None,
        }

    return info


def analyze_attention_head_weights(model: Any, layer_idx: int = 17, head_idx: int = 27) -> dict[str, Any]:
    """
    Analyze the specific weights that H27@L17 uses.
    
    This is the horizon-tracking head identified in MI experiments.
    """
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    hidden_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_dim // num_heads

    result = {
        "layer": layer_idx,
        "head": head_idx,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }

    # Q, K, V projections for this head
    # Weight shape is typically (hidden_dim, hidden_dim) for each projection
    # Each head uses [head_idx * head_dim : (head_idx + 1) * head_dim] slice

    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    if hasattr(attn, 'q_proj'):
        W_q = attn.q_proj.weight.detach()
        result["q_proj"] = {
            "full_shape": list(W_q.shape),
            "head_slice": f"[{start_idx}:{end_idx}, :]",
            "head_weight_shape": [head_dim, hidden_dim],
            "head_weight_norm": float(W_q[start_idx:end_idx, :].norm().item()),
        }

    if hasattr(attn, 'k_proj'):
        W_k = attn.k_proj.weight.detach()
        result["k_proj"] = {
            "full_shape": list(W_k.shape),
            "head_slice": f"[{start_idx}:{end_idx}, :]",
            "head_weight_shape": [head_dim, hidden_dim],
            "head_weight_norm": float(W_k[start_idx:end_idx, :].norm().item()),
        }

    if hasattr(attn, 'v_proj'):
        W_v = attn.v_proj.weight.detach()
        result["v_proj"] = {
            "full_shape": list(W_v.shape),
            "head_slice": f"[{start_idx}:{end_idx}, :]",
            "head_weight_shape": [head_dim, hidden_dim],
            "head_weight_norm": float(W_v[start_idx:end_idx, :].norm().item()),
        }

    if hasattr(attn, 'o_proj'):
        W_o = attn.o_proj.weight.detach()
        result["o_proj"] = {
            "full_shape": list(W_o.shape),
            "head_slice": f"[:, {start_idx}:{end_idx}]",
            "head_weight_shape": [hidden_dim, head_dim],
            "head_weight_norm": float(W_o[:, start_idx:end_idx].norm().item()),
        }

    return result


def main():
    print("OLMo Forward Pass Trace")
    print("=" * 5)

    MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")

    if not MODEL_DIR.exists():
        print(f"Model not found at {MODEL_DIR}")
        print("Showing structure analysis only (no forward trace)")
        return

    print(f"\nLoading model from {MODEL_DIR}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Model config from actual config.json
    print("\n" + "-" * 5)
    print("Model Configuration (from config.json)")
    print("-" * 5)
    config = model.config
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  sliding_window: {getattr(config, 'sliding_window', 'N/A')}")
    print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"  rms_norm_eps: {getattr(config, 'rms_norm_eps', 'N/A')}")
    print(f"  hidden_act: {getattr(config, 'hidden_act', 'N/A')}")

    # Layer types pattern
    layer_types = getattr(config, 'layer_types', None)
    if layer_types:
        full_attn_layers = [i for i, t in enumerate(layer_types) if t == 'full_attention']
        sliding_attn_layers = [i for i, t in enumerate(layer_types) if t == 'sliding_attention']
        print(f"  layer_types pattern: {layer_types[:4]}... (repeats)")
        print(f"  full_attention layers: {full_attn_layers}")
        print(f"  sliding_attention layers: {len(sliding_attn_layers)} layers")

    # Single layer analysis
    print("\n" + "-" * 5)
    print("Layer 17 Structure (peak horizon signal)")
    print("-" * 5)
    layer_info = trace_single_layer(model, 17)
    for comp_name, comp_info in layer_info["components"].items():
        print(f"\n  {comp_name}:")
        for k, v in comp_info.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for k2, v2 in v.items():
                    print(f"      {k2}: {v2}")
            else:
                print(f"    {k}: {v}")

    # H27@L17 weight analysis
    print("\n" + "-" * 5)
    print("H27@L17 Weight Analysis (horizon-tracking head)")
    print("-" * 5)
    head_info = analyze_attention_head_weights(model, layer_idx=17, head_idx=27)
    for k, v in head_info.items():
        if isinstance(v, dict):
            print(f"\n  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    # Forward pass trace
    print("\n" + "-" * 5)
    print("Forward Pass Trace (short prompt)")
    print("-" * 5)

    tracer = OLMoTracer()
    tracer.attach(model)

    test_prompt = "Hello world"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    print(f"  Prompt: '{test_prompt}'")
    print(f"  Token count: {inputs['input_ids'].shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        _ = model(**inputs)  # Run forward pass; hooks capture the trace
    t1 = time.time()

    tracer.detach()

    summary = tracer.summarize()
    print(f"\n  Forward pass time: {(t1-t0)*1000:.1f}ms")
    print(f"  Total module calls: {summary['total_modules_called']}")
    print(f"  Total weight params accessed: {summary['total_weight_params']:,}")

    print("\n  Module types called:")
    for mtype, count in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
        print(f"    {mtype}: {count}")

    print("\n  First 10 Linear operations:")
    for i, op in enumerate(summary["linear_ops"][:10]):
        print(f"    {i+1}. {op['name']}")
        print(f"       Weight: {op['weight_shape']}, In: {op['input']}, Out: {op['output']}")

    print("\nForward pass trace complete.")


if __name__ == "__main__":
    main()
