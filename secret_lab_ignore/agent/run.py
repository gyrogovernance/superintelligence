from __future__ import annotations

import numpy as np

from .runtime import GyroASI, GyroASIConfig


def main() -> None:
    cfg = GyroASIConfig(
        atlas_dir="data/atlas",
        K=16,
        eta=0.01,
        temperature=0.7,
        weight_penalty=0.02,
        seed=42,
    )

    asi = GyroASI(cfg)

    # dummy semantic vectors (replace with real embeddings)
    d = 256 * cfg.K
    rng = np.random.default_rng(123)

    print("Running 100 steps...")
    for step in range(100):
        x_in = rng.standard_normal(d).astype(np.float32)
        b = asi.step(x_in)
        if step < 10 or step % 20 == 0:
            print(f"step {step:3d}: byte={b:3d} (0x{b:02x}) char={chr(b) if 32 <= b < 127 else '.'}")

    print(f"\nGenealogy ({len(asi.genealogy)} bytes):")
    print(bytes(asi.genealogy[:64]))


if __name__ == "__main__":
    main()