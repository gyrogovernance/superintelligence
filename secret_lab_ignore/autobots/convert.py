"""
Structural projection of external model weights.

CLI:
    python -m secret_lab_ignore.autobots.convert --physics-model ... --target-model ... --output ...

Takes a trained HGT physics model and an external model (e.g. Bolmo-1B).
Decomposes external embeddings into:
  E_original = E_structural + E_residual

where E_structural is the component aligned with byte physics.

Usage:
    python -m secret_lab_ignore.autobots.convert \
        --physics-model data/autobots/model \
        --target-model data/models/Bolmo-1B \
        --output data/autobots/bolmo_structural
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from safetensors.torch import save_file, safe_open

from .model import HGTForCausalLM


class StructuralProjector:
    def __init__(self, physics_model: HGTForCausalLM):
        self.physics = physics_model

    def compute_structural_basis(self) -> Tensor:
        """256 byte embeddings from the physics model -> orthonormal basis."""
        with torch.no_grad():
            ids = torch.arange(256, dtype=torch.long).unsqueeze(0)
            families = (torch.bitwise_right_shift(ids ^ self.physics.config.gene_mic_s, 6)) & 0x3
            micro_refs = (ids ^ self.physics.config.gene_mic_s) & 0x3F
            bl1 = self.physics.bl1(ids, families, micro_refs)
            return bl1.squeeze(0)

    def project_token_embeddings(
        self,
        target_embeddings: Tensor,
        target_tokenizer: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Returns (E_structural, E_residual)."""
        basis = self.compute_structural_basis()
        Q, _ = torch.linalg.qr(basis.T)
        E_structural = target_embeddings @ Q @ Q.T
        E_residual = target_embeddings - E_structural
        return E_structural, E_residual

    def apply_to_model(
        self,
        target_model_path: Path,
        output_path: Path,
    ) -> Dict[str, Union[float, str]]:
        """Full conversion pipeline. Returns alignment metrics."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        target_path = Path(target_model_path)
        embed_path = target_path / "model.safetensors"
        if not embed_path.exists():
            embed_path = target_path / "model.safetensors.index.json"
        if embed_path.suffix == ".json":
            return {"error": "Sharded model not fully supported"}

        with safe_open(str(embed_path), framework="pt") as f:
            keys = list(f.keys())
            target_embed = None
            for k in keys:
                if "embed" in k.lower():
                    target_embed = f.get_tensor(k)
                    break
        if target_embed is None:
            return {"error": "No embedding layer found"}

        basis = self.compute_structural_basis()
        if target_embed.shape[1] != basis.shape[1]:
            return {
                "error": (
                    f"dim mismatch: target D={target_embed.shape[1]} "
                    f"vs basis={basis.shape[1]}"
                )
            }

        E_struct, E_res = self.project_token_embeddings(target_embed, None)
        alignment = (E_struct.norm() / (target_embed.norm() + 1e-8)).item()

        report = {"alignment_norm_ratio": alignment}
        with open(output_path / "alignment_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = HGTForCausalLM.from_pretrained(args.physics_model)
    proj = StructuralProjector(model)
    report = proj.apply_to_model(args.target_model, args.output)
    print(report)


if __name__ == "__main__":
    main()
