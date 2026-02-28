"""
Universal, reversible tensor conversion for Bolmo safetensors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file, safe_open

from ..fwht import fwht_1d, fwht_right, is_power_of_two
from .manifest import ResonatorManifest, TensorRecord, TransformSpec


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def choose_transform(shape: list[int], *, convert_1d: bool = False) -> TransformSpec:
    if len(shape) == 2:
        in_dim = int(shape[1])
        if is_power_of_two(in_dim):
            return TransformSpec(type="WHT_R", axis=1, n=in_dim)
        return TransformSpec(type="NONE")
    if len(shape) == 1 and convert_1d:
        n = int(shape[0])
        if is_power_of_two(n):
            return TransformSpec(type="WHT_1D", axis=0, n=n)
        return TransformSpec(type="NONE")
    return TransformSpec(type="NONE")


def apply_transform(tensor: torch.Tensor, spec: TransformSpec) -> torch.Tensor:
    if spec.type == "NONE":
        return tensor
    if spec.type == "WHT_1D":
        return fwht_1d(tensor)
    if spec.type == "WHT_R":
        return fwht_right(tensor)
    raise ValueError(f"Unknown transform type: {spec.type}")


def _include_name(
    name: str,
    include_prefixes: Iterable[str] | None,
    include_names: set[str] | None,
) -> bool:
    if include_names is not None:
        return name in include_names
    if include_prefixes is None:
        return True
    return any(name.startswith(prefix) for prefix in include_prefixes)


def convert_bolmo_safetensors(
    model_dir: Path | str,
    output_dir: Path | str,
    *,
    include_prefixes: tuple[str, ...] | None = ("model.", "lm_head."),
    include_names: set[str] | None = None,
    output_dtype: str = "float16",
    verify_roundtrip: bool = False,
    convert_1d: bool = False,
    max_tensors_per_shard: int | None = None,
) -> tuple[Path, Path]:
    """
    Convert a Bolmo safetensor set into a transformed shard set plus manifest.

    Returns:
        (output_index_json_path, output_manifest_json_path)
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    config_path = model_dir / "config.json"

    if output_dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported output_dtype: {output_dtype}")
    out_dtype = _DTYPE_MAP[output_dtype]

    index_text = index_path.read_text(encoding="utf-8")
    index = json.loads(index_text)
    weight_map: dict[str, str] = index["weight_map"]
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    source_index_hash = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    source_config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest() if config_text else None

    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if _include_name(name, include_prefixes, include_names):
            shard_to_names.setdefault(shard, []).append(name)

    out_weight_map: dict[str, str] = {}
    records: dict[str, TensorRecord] = {}

    for shard_name, names in shard_to_names.items():
        names_sorted = sorted(names)
        if max_tensors_per_shard is not None:
            names_sorted = names_sorted[:max_tensors_per_shard]

        src_shard_path = model_dir / shard_name
        dst_shard_path = output_dir / shard_name
        tensors_out: dict[str, torch.Tensor] = {}

        with safe_open(str(src_shard_path), framework="pt", device="cpu") as reader:
            for name in names_sorted:
                src0 = reader.get_tensor(name)
                orig_dtype = str(src0.dtype).replace("torch.", "").upper()
                src = src0.to(torch.float32)
                shape = list(src.shape)
                spec = choose_transform(shape, convert_1d=convert_1d)
                converted = apply_transform(src, spec)

                if verify_roundtrip and spec.type != "NONE":
                    restored = apply_transform(converted, spec)
                    mae = torch.mean(torch.abs(restored - src)).item()
                    if mae > 1e-4:
                        raise RuntimeError(f"Roundtrip check failed for {name}: mae={mae}")

                tensors_out[name] = converted.to(out_dtype)
                out_weight_map[name] = shard_name
                records[name] = TensorRecord(
                    file=shard_name,
                    orig_shape=shape,
                    orig_dtype=orig_dtype,
                    xform=spec,
                )

        save_file(tensors_out, str(dst_shard_path))

    out_index = {
        "metadata": {
            "source_total_size": index.get("metadata", {}).get("total_size"),
            "source_total_parameters": index.get("metadata", {}).get("total_parameters"),
            "resonator_output_dtype": output_dtype,
        },
        "weight_map": out_weight_map,
    }
    out_index_path = output_dir / "model.safetensors.index.json"
    out_index_path.write_text(json.dumps(out_index, indent=2), encoding="utf-8")

    manifest = ResonatorManifest(
        resonator_version="0.1",
        source_model_dir=str(model_dir),
        source_index_file=str(index_path),
        source_index_hash=source_index_hash,
        source_config_hash=source_config_hash,
        output_dtype=output_dtype,
        wht_kind="hadamard",
        wht_normalization="ortho",
        tensors=records,
    )
    manifest_path = output_dir / "resonator_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    return out_index_path, manifest_path
