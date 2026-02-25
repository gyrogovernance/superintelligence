"""
Manifest helpers for converted resonator tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TransformSpec:
    type: str
    axis: int | None = None
    n: int | None = None


@dataclass
class TensorRecord:
    file: str
    orig_shape: list[int]
    orig_dtype: str
    xform: TransformSpec


@dataclass
class ResonatorManifest:
    resonator_version: str
    source_model_dir: str
    source_index_file: str
    source_index_hash: str | None
    source_config_hash: str | None
    output_dtype: str
    wht_kind: str
    wht_normalization: str
    tensors: dict[str, TensorRecord]

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["tensors"] = {
            name: {
                "file": rec.file,
                "orig_shape": rec.orig_shape,
                "orig_dtype": rec.orig_dtype,
                "xform": asdict(rec.xform),
            }
            for name, rec in self.tensors.items()
        }
        return raw

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ResonatorManifest":
        tensors: dict[str, TensorRecord] = {}
        for name, rec in raw["tensors"].items():
            tensors[name] = TensorRecord(
                file=rec["file"],
                orig_shape=list(rec["orig_shape"]),
                orig_dtype=str(rec["orig_dtype"]),
                xform=TransformSpec(**rec["xform"]),
            )
        return ResonatorManifest(
            resonator_version=str(raw["resonator_version"]),
            source_model_dir=str(raw["source_model_dir"]),
            source_index_file=str(raw["source_index_file"]),
            source_index_hash=raw.get("source_index_hash"),
            source_config_hash=raw.get("source_config_hash"),
            output_dtype=str(raw.get("output_dtype", "float16")),
            wht_kind=str(raw["wht_kind"]),
            wht_normalization=str(raw["wht_normalization"]),
            tensors=tensors,
        )
