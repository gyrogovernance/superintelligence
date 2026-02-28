"""
Complete byte-level inference using Router physics + WHT-converted Bolmo weights.

EXPERIMENTAL: Uses SpectralForward which is not the Bolmo computation graph.
For production, use tuning/operator.RouterOperator (compiled by run_tune.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from ..bolmo_vocab import BolmoVocabSpec, byte_to_base_token
from .spectral_forward import SpectralForward
from ..state_encoder import FullStateEncoder
from .store import ResonatorStore


class RouterInference:
    """
    Complete byte-level inference using Router physics and
    WHT-converted Bolmo weights.
    """

    def __init__(
        self,
        four_layers: Any,
        store: ResonatorStore,
        config: dict[str, Any],
        kernel: Any = None,
        *,
        lm_only: bool = False,
        state_scale: float = 1.0,
        embed_scale: float = 1.0,
    ):
        self.four = four_layers
        self.store = store
        self.config = config
        self.kernel = kernel
        self.encoder = FullStateEncoder(dim=2048)
        self.spectral = SpectralForward(store, config, lm_only=lm_only)
        self._last_byte: int = 0xAA
        self.state_scale = float(state_scale)
        self.embed_scale = float(embed_scale)
        tok_cfg = config.get("tokenizer_config", {}) if isinstance(config.get("tokenizer_config"), dict) else {}
        self.vocab = BolmoVocabSpec(offset=int(tok_cfg.get("vocab_offset", 4) if "vocab_offset" in tok_cfg else 4))

    def _get_regs(self):
        return self.four.regs

    def set_last_byte(self, b: int) -> None:
        self._last_byte = int(b) & 0xFF

    def _byte_embedding_vector(self, *, device: str | torch.device = "cpu") -> torch.Tensor | None:
        key = "model.local_encoder.byte_embedding.weight"
        if not self.store.has(key):
            return None
        weight = self.store.get_tensor(key, device=device).to(torch.float32)
        token_id = byte_to_base_token(self._last_byte, self.vocab)
        if token_id < 0 or token_id >= int(weight.shape[0]):
            return None
        return weight[token_id]

    def next_byte_logits(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """
        Compute next-byte logits from current Router state.
        Returns [vocab_size] logits.
        """
        regs = self._get_regs()

        x_hat = self.encoder.encode(
            l1_state8=regs.l1_state8,
            l2_state16=regs.l2_state16,
            l3_state24=regs.l3_state24,
            l4_O=regs.l4.O,
            l4_E=regs.l4.E,
            l4_parity=regs.l4.parity,
            last_byte=self._last_byte,
            kernel=self.kernel,
            device=device,
        )
        emb = self._byte_embedding_vector(device=device)
        if emb is not None:
            x_hat = (self.state_scale * x_hat) + (self.embed_scale * emb)
        else:
            x_hat = self.state_scale * x_hat

        logits = self.spectral.forward(x_hat)
        return logits

    def next_byte_probs(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Get probability distribution over base bytes (0..255)."""
        logits = self.next_byte_logits(device=device)
        byte_logits = logits[self.vocab.base_start : self.vocab.base_end_exclusive]
        return torch.softmax(byte_logits, dim=0)

    def generate(
        self,
        prompt: bytes,
        max_bytes: int = 100,
        *,
        greedy: bool = True,
        device: str | torch.device = "cpu",
    ) -> bytes:
        """
        Generate bytes autoregressively.
        """
        for b in prompt:
            self.four.ingest_byte(b)
            self.set_last_byte(b)

        output = bytearray()

        for _ in range(max_bytes):
            probs = self.next_byte_probs(device=device)

            if greedy:
                next_byte = int(probs.argmax().item())
            else:
                next_byte = int(
                    torch.multinomial(probs, num_samples=1).item()
                )

            output.append(next_byte)
            self.four.ingest_byte(next_byte)
            self.set_last_byte(next_byte)

            if next_byte == 0:
                break

        return bytes(output)


def load_router_inference(
    resonator_dir: Path | str,
    four_layers: Any,
    config_path: Optional[Path | str] = None,
    *,
    lm_only: bool = False,
    state_scale: float = 1.0,
    embed_scale: float = 1.0,
) -> RouterInference:
    """
    Load ResonatorStore + config and build RouterInference.

    lm_only: Use only lm_head (no transformer layers). Works with profile min.
    """
    import json

    resonator_dir = Path(resonator_dir)
    store = ResonatorStore(resonator_dir)

    if config_path is None:
        config_path = Path("data/models/Bolmo-1B/config.json")
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    return RouterInference(
        four_layers=four_layers,
        store=store,
        config=config,
        lm_only=lm_only,
        state_scale=state_scale,
        embed_scale=embed_scale,
    )
