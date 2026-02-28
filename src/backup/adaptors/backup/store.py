"""
Runtime access to converted resonator tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import safe_open

from .manifest import ResonatorManifest, TransformSpec


class ResonatorStore:
    def __init__(self, tensor_dir: Path | str):
        self.tensor_dir = Path(tensor_dir)
        manifest_path = self.tensor_dir / "resonator_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = ResonatorManifest.from_dict(raw)

    def has(self, tensor_name: str) -> bool:
        return tensor_name in self.manifest.tensors

    def transform_spec(self, tensor_name: str) -> TransformSpec:
        return self.manifest.tensors[tensor_name].xform

    def get_tensor(self, tensor_name: str, *, device: torch.device | str = "cpu") -> torch.Tensor:
        rec = self.manifest.tensors[tensor_name]
        shard_path = self.tensor_dir / rec.file
        with safe_open(str(shard_path), framework="pt", device=str(device)) as reader:
            return reader.get_tensor(tensor_name)
